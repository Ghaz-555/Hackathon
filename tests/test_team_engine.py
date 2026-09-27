"""Independent checks for the team's engine and its stage-2 adapter."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from glasswork.team_engine import GlassBoxModel
from glasswork.team_engine.gradients import output_layer_gradients
from glasswork.team_engine.output import softmax_with_temperature
from server.team_adapter import TeamEngine, TRAIN_TEXT


class TeamEngineTests(unittest.TestCase):
    def test_original_source_forward_update_and_generation_parity(self):
        # This fixture was produced by running the untouched Downloads/Hackathon backend.
        expected = json.loads((Path(__file__).parent/'fixtures/team_source_reference.json').read_text())
        model = GlassBoxModel(TRAIN_TEXT, seed=42)
        actual = model.forward('the ca', temperature=.8)
        self.assertEqual(actual['tokens'], expected['tokens'])
        for field in ['combined_embeddings','attention_output','logits']:
            np.testing.assert_allclose(actual[field], expected[field], atol=1e-14, rtol=1e-14)
        for a, b in zip(actual['attention_weights'], expected['attention_weights']):
            np.testing.assert_allclose(a, b, atol=1e-14, rtol=1e-14)
        for c, value in expected['probabilities'].items():
            self.assertAlmostEqual(actual['probabilities'][c], value, places=14)
        update = model.train_step('the ca', 't', .1)
        np.testing.assert_allclose(model.output_weights, expected['output_after_step'], atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(update['d_hidden_state'], expected['d_hidden_state'], atol=1e-14, rtol=1e-14)
        self.assertAlmostEqual(update['loss'], expected['loss_after_step'], places=14)
        self.assertEqual(model.generate('the ',length=20,temperature=.8,seed=7),expected['generation'])

    def test_all_output_and_hidden_gradients_against_finite_differences(self):
        model = GlassBoxModel(TRAIN_TEXT)
        result = model.forward('the ca')
        hidden = result['attention_output'][-1]
        p = np.array([result['probabilities'][model.id_to_character[i]] for i in range(len(model.vocabulary))])
        target = model.vocabulary['t']
        _, dw, dh = output_layer_gradients(hidden,p,target,model.output_weights)
        eps, worst = 1e-5, 0.
        for index in np.ndindex(model.output_weights.shape):
            original = model.output_weights[index]
            model.output_weights[index] = original + eps
            plus = model.calculate_loss('the ca','t')
            model.output_weights[index] = original - eps
            minus = model.calculate_loss('the ca','t')
            model.output_weights[index] = original
            error = abs(dw[index] - (plus-minus)/(2*eps))
            worst = max(worst,error)
            self.assertLess(error, 1e-8)
        def hidden_loss(h):
            logits=h @ model.output_weights
            shifted=logits-logits.max()
            return np.log(np.exp(shifted).sum())-shifted[target]
        for i in range(len(hidden)):
            delta=np.zeros_like(hidden);delta[i]=eps
            self.assertAlmostEqual(dh[i],(hidden_loss(hidden+delta)-hidden_loss(hidden-delta))/(2*eps),places=8)
        print(f'Team gradients: 40 output weights + 4 hidden values; max output error={worst:.3e}')

    def test_learning_changes_only_declared_output_parameters(self):
        engine=TeamEngine()
        model=engine.model
        before={k:v.copy() for k,v in model.params.items()}
        initial=model.calculate_loss('the ca','t')
        for _ in range(100):
            model.train_step('the ca','t')
        self.assertLess(model.calculate_loss('the ca','t'), initial)
        for name,value in model.params.items():
            if name=='output_weights':
                self.assertFalse(np.array_equal(value,before[name]))
            else:
                np.testing.assert_array_equal(value,before[name])
        self.assertEqual(engine.metadata()['trainable_parameter_count'],40)
        self.assertEqual(engine.metadata()['parameter_count'],640)

    def test_trace_preserves_source_attention_and_causality(self):
        engine=TeamEngine()
        raw=engine.model.forward('the ca')
        trace=engine.inspect('the ca',1)['trace']
        dense=np.asarray(trace['blocks'][0]['attention'])[0,0]
        np.testing.assert_allclose(dense.sum(axis=-1),1,atol=1e-15)
        np.testing.assert_array_equal(np.triu(dense,1),np.zeros_like(dense))
        for i,row in enumerate(raw['attention_weights']):
            np.testing.assert_array_equal(dense[i,:i+1],row)
        before=engine.model.forward('the cat')['attention_output'][:3]
        after=engine.model.forward('the mat')['attention_output'][:3]
        np.testing.assert_array_equal(before,after)
        json.dumps(trace,allow_nan=False)

    def test_local_initialization_seed_and_normalized_training(self):
        np.random.seed(12)
        expected=np.random.random(3)
        np.random.seed(12)
        a=GlassBoxModel(TRAIN_TEXT,seed=42)
        np.testing.assert_array_equal(np.random.random(3),expected)
        b=GlassBoxModel(TRAIN_TEXT.upper(),seed=42)
        np.testing.assert_array_equal(a.forward('the ca')['logits'], b.forward('THE CA')['logits'])
        a.train_epoch(TRAIN_TEXT)
        b.train_epoch(TRAIN_TEXT.upper())
        np.testing.assert_array_equal(a.output_weights,b.output_weights)
        for text in ('', 'x'):
            with self.assertRaises(ValueError):
                a.forward(text)
        for text in ('', 't'):
            with self.assertRaises(ValueError):
                a.evaluate(text)

    def test_greedy_temperature_and_extreme_logits_are_stable(self):
        logits=np.array([1000.,0.,-1000.])
        np.testing.assert_array_equal(softmax_with_temperature(logits,0),[1,0,0])
        for t in (1e-300,.3,1,2):
            self.assertTrue(np.isfinite(softmax_with_temperature(logits,t)).all())
            self.assertAlmostEqual(softmax_with_temperature(logits,t).sum(),1)
        for t in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                softmax_with_temperature(logits,t)

    def test_checkpoint_restores_exact_next_update_and_generation(self):
        engine=TeamEngine()
        for _ in range(5):engine.train_step()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'team.npz'
            engine.save(path)
            restored=TeamEngine.load(path)
            self.assertEqual(engine.fingerprint(),restored.fingerprint())
            self.assertEqual(engine.generate('the ',20,.8,7),restored.generate('the ',20,.8,7))
            self.assertEqual(engine.train_step(),restored.train_step())
            self.assertEqual(engine.fingerprint(),restored.fingerprint())
            self.assertEqual(engine.metrics(),restored.metrics())
            with np.load(path,allow_pickle=False) as archive:
                contents={key:archive[key].copy() for key in archive.files}
            contents['output_weights'][0,0]=np.nan
            np.savez(path,**contents)
            with self.assertRaises(ValueError):TeamEngine.load(path)

    def test_3d_trace_reconstructs_each_displayed_operation(self):
        engine = TeamEngine.load()
        model = engine.model
        data = engine.inspect('the cat', .8)
        trace = data['trace']
        self.assertEqual(trace['schema_version'], 4)
        combined = np.asarray(trace['token_embeddings']) + np.asarray(trace['position_embeddings'])
        np.testing.assert_array_equal(combined, trace['embedding_sum'][0])
        for field, weight, captured in [('queries',model.W_q,'query_weights'),('keys',model.W_k,'key_weights'),('values',model.W_v,'value_weights')]:
            np.testing.assert_array_equal(trace[captured], weight)
            np.testing.assert_array_equal(combined @ np.asarray(trace[captured]), trace[field])
        q,k,v = [np.asarray(trace[key]) for key in ('queries','keys','values')]
        attention = np.asarray(trace['blocks'][0]['attention'])[0,0]
        for i in range(len(q)):
            scores = q[i] @ k[:i+1].T / np.sqrt(model.embedding_size)
            expected = np.exp(scores-scores.max()); expected /= expected.sum()
            np.testing.assert_allclose(attention[i,:i+1],expected,atol=1e-15)
        hidden = attention @ v
        np.testing.assert_allclose(hidden,trace['blocks'][0]['head_outputs'][0][0],atol=1e-15)
        np.testing.assert_allclose(hidden[-1] @ np.asarray(trace['output_weights']),data['logits'],atol=1e-15)
        np.testing.assert_allclose(softmax_with_temperature(np.array(data['logits']),.8),data['probabilities'])

    def test_capture_is_read_only_and_does_not_change_predictions(self):
        model = GlassBoxModel(TRAIN_TEXT)
        before = {k:v.copy() for k,v in model.params.items()}
        raw = model.forward('the ca')
        traced = model.forward('the ca',capture_trace=True)
        np.testing.assert_array_equal(raw['logits'],traced['logits'])
        traced['internals']['output_weights'].fill(0)
        for key,value in before.items():
            np.testing.assert_array_equal(value,model.params[key])

    def test_request_limits_include_normalized_context(self):
        engine=TeamEngine()
        result=engine.inspect('THE CA',1)
        self.assertEqual(result['prompt'],'the ca')
        self.assertEqual(len(engine.inspect('t'*128,1)['trace']['tokens']),128)
        with self.assertRaises(ValueError):engine.inspect('t'*129,1)
        with self.assertRaises(ValueError):engine.generate('the',-1,1,7)


if __name__=='__main__':unittest.main()
