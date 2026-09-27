"""The larger model uses the same real training path and exposes all layers."""
import unittest
import numpy as np
from fastapi.testclient import TestClient
from server.app import create_app
from server.transformer_adapter import TransformerEngine

class TransformerApiTests(unittest.TestCase):
    def test_trace_reconstructs_logits_and_all_parameters_learn(self):
        engine = TransformerEngine.load().reset()
        before = {k: v.copy() for k, v in engine.model.params.items()}
        engine.train_step()
        for name in before:
            self.assertFalse(np.array_equal(before[name], engine.model.params[name]), name)
        result = engine.inspect('the cat', .7)
        trace = result['trace']
        np.testing.assert_allclose(np.array(trace['final_hidden']) @ trace['output_weights'] + trace['output_bias'], result['logits'])
        self.assertEqual(len(trace['blocks']), 2)
        self.assertEqual(np.shape(trace['blocks'][0]['attention']), (1, 2, 7, 7))
        self.assertEqual(engine.metadata()['parameter_count'], 28186)

    def test_checkpoint_quality_and_repeatable_generation(self):
        engine = TransformerEngine.load()
        metrics = engine.metrics()
        self.assertLess(metrics['validation'], 1.0)
        sample = engine.generate('the cat ', 64, .7, 7)
        self.assertEqual(sample, engine.generate('the cat ', 64, .7, 7))
        self.assertIn('sleeps', sample)
        self.assertIn('mat', sample)
        self.assertGreater(engine.reset().metrics()['validation'], metrics['validation'])

    def test_session_reset_restore_and_context_validation(self):
        client = TestClient(create_app())
        state = client.post('/api/sessions', json={'engine':'transformer'}).json()
        url = '/api/sessions/'+state['session_id']
        self.assertEqual(state['trainable_parameter_count'], 28186)
        self.assertEqual(client.post(url+'/inspect', json={'prompt':'a'*33}).status_code, 422)
        random = client.post(url+'/reset', json={'revision':0,'mode':'random'}).json()
        self.assertEqual(random['step'], 0)
        trained = client.post(url+'/train', json={'revision':1, 'prompt':'the cat', 'steps':1}).json()['after']
        self.assertEqual(trained['step'], 1)
        restored = client.post(url+'/reset', json={'revision':2,'mode':'trained'}).json()
        self.assertEqual(restored['fingerprint'], state['fingerprint'])
        self.assertEqual(client.post('/api/sessions', json={'engine':'unknown'}).status_code, 422)
