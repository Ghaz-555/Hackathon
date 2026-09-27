"""Contract and integration tests against the real checkpoint and NumPy engine."""
from concurrent.futures import ThreadPoolExecutor
import unittest
import numpy as np
from fastapi.testclient import TestClient
from server.team_adapter import TeamEngine
from glasswork.team_engine.output import softmax_with_temperature as distribution
from server.app import create_app, ROOT


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())
        response = self.client.post('/api/sessions', json={})
        self.assertEqual(response.status_code, 201)
        self.state = response.json()
        self.url = '/api/sessions/' + self.state['session_id']

    def inspect(self, **kwargs):
        return self.client.post(self.url + '/inspect', json={'prompt':'the ', **kwargs})

    def test_api_matches_engine_and_temperature_preserves_weights(self):
        engine = TeamEngine.load()
        logits = engine.model.forward('the ')['logits']
        low, high = self.inspect(temperature=.3).json(), self.inspect(temperature=1.8).json()
        np.testing.assert_array_equal(low['logits'], logits)
        np.testing.assert_allclose(low['probabilities'], distribution(logits, .3))
        self.assertEqual(low['fingerprint'], high['fingerprint'])
        self.assertEqual(low['step'], high['step'])
        self.assertGreater(high['entropy'], low['entropy'])
        self.assertEqual(len(low['trace']['blocks']), 1)

    def test_training_changes_only_its_session_and_stale_mutation_is_rejected(self):
        other = self.client.post('/api/sessions', json={}).json()
        reset = self.client.post(self.url + '/reset', json={'revision':0, 'mode':'random'}).json()
        self.assertEqual(reset['step'], 0)
        before = self.inspect().json()
        trained = self.client.post(self.url + '/train', json={'revision':1, 'steps':2, 'prompt':'the '})
        self.assertEqual(trained.status_code, 200)
        result = trained.json()
        self.assertEqual(result['before']['fingerprint'], before['fingerprint'])
        self.assertNotEqual(result['after']['fingerprint'], before['fingerprint'])
        self.assertEqual(result['after']['step'], 2)
        self.assertEqual(len(result['after']['history']), 3)
        other_state = self.client.get('/api/sessions/' + other['session_id']).json()
        self.assertEqual(other_state['fingerprint'], other['fingerprint'])
        stale = self.client.post(self.url + '/train', json={'revision':1})
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(self.client.get(self.url).json()['step'], 2)

    def test_concurrent_mutations_cannot_both_use_same_revision(self):
        def mutate(_):
            return self.client.post(self.url + '/train', json={'revision':0}).status_code
        with ThreadPoolExecutor(max_workers=2) as executor:
            self.assertEqual(sorted(executor.map(mutate, range(2))), [200,409])
        self.assertEqual(self.client.get(self.url).json()['step'], self.state['step'] + 1)

    def test_seeded_sampling_and_reset_restore_checkpoint_exactly(self):
        request = {'prompt':'the ', 'temperature':.8, 'seed':7, 'count':30}
        a = self.client.post(self.url + '/sample', json=request).json()
        b = self.client.post(self.url + '/sample', json=request).json()
        self.assertEqual(a['text'], b['text'])
        self.client.post(self.url + '/reset', json={'revision':0, 'mode':'random'})
        restored = self.client.post(self.url + '/reset', json={'revision':1, 'mode':'trained'}).json()
        self.assertEqual(restored['fingerprint'], self.state['fingerprint'])
        self.assertEqual(restored['step'], self.state['step'])

    def test_bad_prompt_does_not_mutate_weights_and_limits_are_enforced(self):
        for body in ({'prompt':'HELLO'}, {'prompt':''}, {'prompt':'a'*129}, {'temperature':-1}, {'temperature':3}):
            self.assertEqual(self.client.post(self.url + '/inspect', json=body).status_code, 422)
        for body in ({'revision':0,'prompt':'HELLO'}, {'revision':0,'steps':100}):
            self.assertEqual(self.client.post(self.url + '/train', json=body).status_code, 422)
        self.assertEqual(self.client.get(self.url).json()['fingerprint'], self.state['fingerprint'])
        self.assertEqual(self.client.post(self.url+'/sample', json={'count':1000}).status_code, 422)

    def test_session_capacity_deletion_and_expiry(self):
        client = TestClient(create_app(max_sessions=1))
        first = client.post('/api/sessions', json={}).json()['session_id']
        self.assertEqual(client.post('/api/sessions', json={}).status_code, 503)
        self.assertEqual(client.delete('/api/sessions/'+first).status_code, 204)
        self.assertEqual(client.get('/api/sessions/'+first).status_code, 404)
        self.assertEqual(client.post('/api/sessions', json={}).status_code, 201)
        expiring = TestClient(create_app(ttl=-1))
        sid = expiring.post('/api/sessions', json={}).json()['session_id']
        self.assertEqual(expiring.get('/api/sessions/'+sid).status_code, 404)

    def test_health_and_missing_checkpoint(self):
        self.assertEqual(self.client.get('/api/health').json()['engine'], 'numpy')
        missing = TestClient(create_app(checkpoint=ROOT/'missing.npz'))
        self.assertEqual(missing.post('/api/sessions', json={}).status_code, 503)


if __name__ == '__main__':
    unittest.main()
