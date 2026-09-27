"""Independent numerical and behavioral checks, not just shape smoke tests."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from glasswork import (CharacterTokenizer, Config, TinyTransformer, Trainer,
                       load_checkpoint, save_checkpoint)
from glasswork.math import cross_entropy, layer_norm, softmax
from glasswork.model import distribution, trace_to_json
from glasswork.training import Adam, AdamConfig, evaluate


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(vocab_size=5, context_length=4, d_model=4,
                             n_heads=2, n_layers=1, d_ff=6, seed=12)
        self.model = TinyTransformer(self.config)
        self.x = np.array([[0, 1, 0], [2, 1, 3]])  # repeated embeddings must accumulate
        self.y = np.array([[1, 0, 2], [1, 3, 4]])

    def test_every_parameter_gradient_against_central_difference(self):
        _, grads = self.model.loss_and_gradients(self.x, self.y)
        self.assertEqual(grads.keys(), self.model.params.keys())
        epsilon, worst, checked = 1e-5, 0.0, 0
        for name, parameter in self.model.params.items():
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = self.model.loss(self.x, self.y)
                parameter[index] = original - epsilon
                minus = self.model.loss(self.x, self.y)
                parameter[index] = original
                numerical = (plus - minus) / (2 * epsilon)
                analytical = grads[name][index]
                error = abs(numerical - analytical)
                worst = max(worst, error)
                self.assertLessEqual(error, 2e-6 + 2e-4 * abs(numerical),
                                     f"{name}{index}: analytic={analytical}, numerical={numerical}")
                checked += 1
        print(f"\nCentral differences: {checked} parameters; max absolute error={worst:.3e}")

    def test_two_layer_gradients_across_residual_paths(self):
        model = TinyTransformer(replace(self.config, n_layers=2))
        _, grads = model.loss_and_gradients(self.x, self.y)
        rng = np.random.default_rng(99)
        for name, p in model.params.items():
            # A random-direction derivative checks the entire parameter array at once.
            direction = rng.normal(size=p.shape)
            direction /= np.linalg.norm(direction)
            original = p.copy()
            p[...] = original + 1e-5 * direction
            plus = model.loss(self.x, self.y)
            p[...] = original - 1e-5 * direction
            minus = model.loss(self.x, self.y)
            p[...] = original
            np.testing.assert_allclose(np.sum(grads[name] * direction), (plus - minus) / 2e-5,
                                       atol=2e-6, rtol=2e-4, err_msg=name)

    def test_future_tokens_cannot_change_prefix_logits(self):
        original = self.model.forward([[0, 1, 2, 3]]).logits
        changed = self.model.forward([[0, 1, 4, 0]]).logits
        np.testing.assert_array_equal(original[:, :2], changed[:, :2])
        shorter = self.model.forward([[0, 1]]).logits
        np.testing.assert_allclose(original[:, :2], shorter, atol=1e-14, rtol=0)

    def test_attention_trace_is_real_normalized_and_causal(self):
        result = self.model.forward(self.x, capture_trace=True)
        trace = result.trace
        for block in trace["blocks"]:
            a = block["attention"]
            np.testing.assert_allclose(a.sum(axis=-1), 1, atol=1e-15)
            self.assertTrue((a >= 0).all())
            self.assertTrue((a[..., ~trace["causal_mask"]] == 0).all())
            np.testing.assert_allclose(block["head_outputs"], a @ block["value"], atol=1e-15)
            expected = softmax(np.where(trace["causal_mask"], block["scaled_scores"], -np.inf))
            np.testing.assert_array_equal(a, expected)
            np.testing.assert_array_equal(block["output"], block["attention_residual"] + block["ff_projection"])
        np.testing.assert_array_equal(trace["logits"], result.logits)
        np.testing.assert_allclose(trace["probabilities"], softmax(result.logits), atol=1e-15)
        json.dumps(trace_to_json(trace), allow_nan=False)
        # UI manipulation of a trace must not be a back door for mutating the model.
        before = self.model.forward(self.x).logits.copy()
        trace["token_embeddings"].fill(100)
        trace["logits"].fill(-100)
        np.testing.assert_array_equal(before, self.model.forward(self.x).logits)

    def test_logits_and_loss_are_stable_at_large_magnitudes(self):
        logits = np.array([[[1000., 0., -1000.]]])
        loss, grad = cross_entropy(logits, np.array([[2]]))
        self.assertEqual(loss, 2000.)
        np.testing.assert_array_equal(grad, [[[1., 0., -1.]]])
        np.testing.assert_allclose(softmax(logits).sum(axis=-1), 1)

    def test_layer_norm_constant_input_and_shift_invariance(self):
        x = np.array([[[2., 2., 2., 2.], [1., 2., 3., 4.]]])
        gain, bias = np.ones(4), np.zeros(4)
        out, _ = layer_norm(x, gain, bias)
        shifted, _ = layer_norm(x + 100, gain, bias)
        np.testing.assert_allclose(out, shifted, atol=1e-13)
        np.testing.assert_array_equal(out[0, 0], np.zeros(4))
        self.assertTrue(np.isfinite(out).all())

    def test_temperature_changes_distribution_without_changing_weights(self):
        before = {k: p.copy() for k, p in self.model.params.items()}
        logits = np.array([1., 2., 4.])
        cold, hot = distribution(logits, .5), distribution(logits, 1.5)
        entropy = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0]))
        self.assertLess(entropy(cold), entropy(hot))
        np.testing.assert_array_equal(distribution(logits, 0), [0, 0, 1])
        for temperature in (1e-300, .5, 1, 1e300):
            p = distribution(logits, temperature)
            self.assertTrue(np.isfinite(p).all())
            self.assertAlmostEqual(p.sum(), 1)
        a = self.model.generate([0, 1], max_new_tokens=10, seed=3)
        b = self.model.generate([0, 1], max_new_tokens=10, seed=3)
        np.testing.assert_array_equal(a, b)
        for k in before:
            np.testing.assert_array_equal(before[k], self.model.params[k])

    def test_seed_reproducibility_and_batch_independence(self):
        same = TinyTransformer(self.config)
        np.testing.assert_array_equal(self.model.forward(self.x).logits, same.forward(self.x).logits)
        result = self.model.forward(self.x).logits
        for i in range(len(self.x)):
            np.testing.assert_allclose(result[i], self.model.forward(self.x[i:i+1]).logits[0], atol=1e-14)

    def test_tiny_dataset_can_be_overfit(self):
        model = TinyTransformer(Config(vocab_size=2, context_length=8, d_model=8,
                                       n_heads=2, n_layers=1, d_ff=16))
        trainer = Trainer(model, Adam(model.params, AdamConfig(learning_rate=.01)))
        stream = np.array([0, 1] * 32)
        initial = evaluate(model, stream)
        for _ in range(100):
            trainer.train_step(stream, batch_size=4)
        final = evaluate(model, stream)
        self.assertLess(final, .02)
        self.assertLess(final, initial * .05)
        generated = model.generate([0], max_new_tokens=15, temperature=0)
        np.testing.assert_array_equal(generated, [0, 1] * 8)
        print(f"Overfit check: loss {initial:.6f} -> {final:.6f}")

    def test_adam_first_step_matches_hand_calculation(self):
        params = {"w": np.array([1., -1.])}
        cfg = AdamConfig(learning_rate=.1, max_grad_norm=100)
        opt = Adam(params, cfg)
        g = np.array([2., -3.])
        opt.step(params, {"w": g})
        np.testing.assert_allclose(params["w"], np.array([1., -1.]) - .1 * g / (np.abs(g) + cfg.epsilon))
        np.testing.assert_allclose(opt.m["w"], .1 * g)
        np.testing.assert_allclose(opt.v["w"], .001 * g**2)

    def test_clipping_and_bad_gradient_updates_are_atomic(self):
        params = {"a": np.array([1.]), "b": np.array([2.])}
        opt = Adam(params)
        metrics = opt.step(params, {"a": np.array([3.]), "b": np.array([4.])})
        self.assertAlmostEqual(metrics["gradient_norm"], 5.)
        self.assertAlmostEqual(metrics["clip_scale"], .2)
        np.testing.assert_allclose(opt.m["a"], [.06])
        before = {k: v.copy() for k, v in params.items()}
        with self.assertRaises(FloatingPointError):
            opt.step(params, {"a": np.array([1.]), "b": np.array([np.nan])})
        self.assertEqual(opt.step_count, 1)
        for k in params:
            np.testing.assert_array_equal(params[k], before[k])

    def test_checkpoint_restores_predictions_and_exact_next_training_step(self):
        tok = CharacterTokenizer("abcde")
        trainer = Trainer(self.model)
        tokens = tok.encode("abcdeabcdeabcde")
        for _ in range(3):
            trainer.train_step(tokens, batch_size=2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            save_checkpoint(path, trainer, tok, metadata={"purpose": "test"})
            restored, tokenizer, meta = load_checkpoint(path)
            self.assertEqual(tokenizer.characters, tok.characters)
            self.assertEqual(meta, {"purpose": "test"})
            np.testing.assert_array_equal(trainer.model.forward(self.x).logits,
                                          restored.model.forward(self.x).logits)
            self.assertEqual(trainer.train_step(tokens, batch_size=2), restored.train_step(tokens, batch_size=2))
            for k in self.model.params:
                np.testing.assert_array_equal(trainer.model.params[k], restored.model.params[k])
                np.testing.assert_array_equal(trainer.optimizer.m[k], restored.optimizer.m[k])
                np.testing.assert_array_equal(trainer.optimizer.v[k], restored.optimizer.v[k])

    def test_checkpoint_rejects_corruption_and_unknown_versions(self):
        tok = CharacterTokenizer("abcde")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            save_checkpoint(path, Trainer(self.model), tok)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in archive.files}
            key = "param/token_embedding"
            original = arrays[key].copy()
            arrays[key][0, 0] = np.nan
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                load_checkpoint(path)
            arrays[key] = original
            header = json.loads(str(arrays["header"].item()))
            header["format_version"] = 999
            arrays["header"] = np.array(json.dumps(header))
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_evaluation_and_generation_do_not_change_training_rng(self):
        trainer = Trainer(self.model)
        before = json.dumps(trainer.rng.bit_generator.state)
        evaluate(self.model, np.array([0, 1, 2, 3, 4, 0]))
        self.model.generate([0], max_new_tokens=4)
        self.assertEqual(before, json.dumps(trainer.rng.bit_generator.state))

    def test_evaluation_weights_partial_final_chunk_by_target_count(self):
        tokens = np.array([0, 1, 2, 3, 4, 0, 1])
        first = self.model.loss(tokens[:4][None, :], tokens[1:5][None, :])
        final = self.model.loss(tokens[4:6][None, :], tokens[5:7][None, :])
        self.assertAlmostEqual(evaluate(self.model, tokens), (4 * first + 2 * final) / 6)

    def test_single_token_and_full_context_backward_are_finite_and_read_only(self):
        before = {k: p.copy() for k, p in self.model.params.items()}
        for length in (1, self.config.context_length):
            ids = np.arange(length)[None, :]
            loss, grads = self.model.loss_and_gradients(ids, (ids + 1) % self.config.vocab_size)
            self.assertTrue(np.isfinite(loss))
            self.assertTrue(all(np.isfinite(g).all() for g in grads.values()))
        for k in before:
            np.testing.assert_array_equal(before[k], self.model.params[k])

    def test_dataset_split_has_no_duplicate_sentences_and_known_characters(self):
        root = Path(__file__).resolve().parents[1]
        train = (root / "data/train.txt").read_text()
        validation = (root / "data/validation.txt").read_text()
        self.assertFalse(set(train.splitlines()) & set(validation.splitlines()))
        tokenizer = CharacterTokenizer.from_text(train)
        self.assertGreater(len(tokenizer.encode(validation)), self.config.context_length)

    def test_tokenizer_unicode_round_trip_and_unknown_rejection(self):
        text = "aé🍁 a"
        tokenizer = CharacterTokenizer.from_text(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.decode(tokenizer.encode("")), "")
        with self.assertRaises(ValueError):
            tokenizer.encode("z")
        with self.assertRaises(ValueError):
            tokenizer.decode([-1])

    def test_invalid_inputs_fail_clearly(self):
        for ids in ([[5]], [[-1]], [[1.5]], [[]], [0, 1], [[0] * 5], [[True]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.model.forward(ids)
        for temp in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                distribution(np.zeros(3), temp)
        with self.assertRaises(ValueError):
            self.model.loss(self.x, np.array([[0, 1]]))
        with self.assertRaises(ValueError):
            Config(vocab_size=2, d_model=3, n_heads=2)
        with self.assertRaises(ValueError):
            Trainer(self.model).train_step(np.array([0, 1]))


if __name__ == "__main__":
    unittest.main()
