"""Export a reproducible GPT-2 token-embedding slice using NumPy only.

Downloads just the safetensors header and wte.weight byte range. The browser gets
raw float32 vectors for cosine search plus a three-component PCA projection.
"""
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache'
OUT = ROOT / 'public' / 'embeddings'
REVISION = '607a30d783dfa663caf39e06633721c8d4cfcd7e'
BASE = f'https://huggingface.co/openai-community/gpt2/resolve/{REVISION}'
WORDS = '''king queen man woman boy girl father mother brother sister son daughter
prince princess husband wife uncle aunt cat dog horse bird fish lion tiger wolf
apple orange banana fruit food bread water coffee tea milk red blue green yellow
black white purple pink color happy sad angry calm joy love hate fear music song
piano guitar violin drum art artist painting science physics chemistry biology
math algebra geometry number vector matrix neuron brain model language word token
computer software hardware code program data network machine learning human robot
sun moon earth mars venus planet star space galaxy universe light dark day night
hot cold warm cool fire ice snow rain winter summer spring autumn north south east
west city town country village capital France Paris Germany Berlin Italy Rome
Japan Tokyo China Beijing Canada Toronto Ottawa England London Spain Madrid
America American Europe European French German Italian Japanese Chinese Canadian
walk walking walked run running ran swim swimming write writing wrote read reading
book paper school student teacher university education doctor nurse hospital health
car truck train bus plane boat road street river ocean mountain forest tree flower
money bank dollar euro rich poor buy sell price market work job business company
small large big tiny huge short long fast slow young old good bad better best
one two three four five six seven eight nine ten first second third zero hundred
truth false true fact idea thought reason question answer knowledge memory dream
chess game play player ball team sport football basketball tennis family friend
freedom justice law court judge government democracy citizen public private home'''.split()


def get_bytes(url, start=None, end=None):
    # A query key avoids stale CDN range responses being reused for another range.
    if start is not None:
        url += f'?range={start}-{end}'
    headers = {'User-Agent': 'Glasswork-embedding-demo/1.0'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{end}'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
        if start is not None:
            expected = end-start+1
            if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                raise RuntimeError('Server did not honor the requested byte range')
            data = response.read(expected+1)
            if len(data) != expected:
                raise RuntimeError('Incomplete embedding download')
            return data
        return response.read()


def main():
    CACHE.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    vocab_file = CACHE / f'{REVISION}-vocab.json'
    if not vocab_file.exists():
        vocab_file.write_bytes(get_bytes(BASE+'/vocab.json'))
    vocab = json.loads(vocab_file.read_text())
    embedding_file = CACHE / f'{REVISION}-wte.npy'
    if not embedding_file.exists():
        url = BASE+'/model.safetensors'
        header_size = struct.unpack('<Q', get_bytes(url, 0, 7))[0]
        if header_size > 2_000_000:
            raise RuntimeError('Unexpected safetensors header size')
        header = json.loads(get_bytes(url, 8, 7+header_size))
        tensor = header['wte.weight']
        assert tensor['dtype'] == 'F32' and tensor['shape'] == [50257, 768], tensor
        start, end = tensor['data_offsets']
        print('Downloading GPT-2 input embeddings (154 MB)…', flush=True)
        raw = get_bytes(url, 8+header_size+start, 8+header_size+end-1)
        weights = np.frombuffer(raw, dtype='<f4').reshape(tensor['shape'])
        np.save(embedding_file, weights)
    weights = np.load(embedding_file, mmap_mode='r')
    # Leading-space alphabetic tokens keep the same tokenization convention.
    # The preset words are guaranteed inclusion, not given special positions.
    selected = {vocab['Ġ'+word] for word in WORDS if 'Ġ'+word in vocab}
    for token, token_id in sorted(vocab.items(), key=lambda pair: pair[1]):
        if len(selected) >= 2048:
            break
        if re.fullmatch(r'Ġ[A-Za-z]{2,18}', token):
            selected.add(token_id)
    ids = sorted(selected)
    inverse = {i: token for token, i in vocab.items()}
    vectors = np.array(weights[ids], dtype='<f4')
    mean = vectors.mean(axis=0, dtype=np.float64)
    centered = vectors.astype(np.float64)-mean
    print('Computing PCA and exact 768-dimensional reference rankings…', flush=True)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T@centered)
    basis = eigenvectors[:, -3:][:, ::-1].copy()
    # Fix eigenvector signs so regenerated plots do not randomly mirror axes.
    for axis in range(3):
        if basis[np.argmax(np.abs(basis[:, axis])), axis] < 0:
            basis[:, axis] *= -1
    projected = centered@basis
    scale = 8/np.percentile(np.linalg.norm(projected, axis=1), 95)
    points = projected*scale
    raw = vectors.tobytes()
    (OUT/'vectors.f32').write_bytes(raw)
    tokens = [{'id':i, 'label':inverse[i][1:], 'token':inverse[i], 'position':point.tolist()} for i, point in zip(ids, points)]
    lookup = {t['label']:i for i,t in enumerate(tokens)}
    a,b,c = (lookup[word] for word in ['king','man','woman'])
    query = vectors[a]-vectors[b]+vectors[c]
    scores = (vectors@query)/(np.linalg.norm(vectors,axis=1)*np.linalg.norm(query))
    scores[[a,b,c]] = -np.inf
    ranking = np.argsort(-scores)
    examples = [{'label':tokens[i]['label'], 'cosine':float(scores[i])} for i in ranking[:10]]
    meta = {
        'schemaVersion':1, 'model':'GPT-2 small', 'repository':'openai-community/gpt2',
        'revision':REVISION, 'source':f'https://huggingface.co/openai-community/gpt2/tree/{REVISION}',
        'tensor':'wte.weight', 'dimensions':768, 'fullVocabularySize':50257,
        'count':len(ids), 'binary':'vectors.f32', 'sha256':hashlib.sha256(raw).hexdigest(),
        'dtype':'float32 little-endian', 'selection':'Preset words, then leading-space alphabetic tokens in token-ID order, up to 2048.',
        'projection':{'method':'PCA on unnormalized input embeddings, centered over this subset',
            'variance':(eigenvalues[-3:][::-1]/eigenvalues.sum()).tolist(),
            'scale':float(scale), 'mean':mean.tolist(), 'basis':basis.T.tolist()},
        'tokens':tokens, 'referenceAnalogy':{'expression':'king - man + woman', 'neighbors':examples,
            'queenRank':int(np.where(ranking==lookup['queen'])[0][0])+1},
    }
    (OUT/'manifest.json').write_text(json.dumps(meta, separators=(',',':'))+'\n')
    print(json.dumps({'count':len(ids),'variance':meta['projection']['variance'],'analogy':meta['referenceAnalogy']},indent=2), flush=True)

if __name__ == '__main__':
    main()
