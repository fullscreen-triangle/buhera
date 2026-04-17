export function computeSEntropy(text) {
    const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
    if (words.length === 0) return { sk: 0, st: 0, se: 0, address: '000' };

    const chars = text.toLowerCase().replace(/\s/g, '');
    const freq = {};
    for (const c of chars) freq[c] = (freq[c] || 0) + 1;
    const n = chars.length || 1;
    let H = 0;
    for (const c in freq) {
        const p = freq[c] / n;
        if (p > 0) H -= p * Math.log2(p);
    }
    const sk = Math.min(H / Math.log2(26), 1.0);

    const temporalMarkers = ['when', 'before', 'after', 'during', 'now', 'then',
        'yesterday', 'today', 'recently', 'previously', 'last', 'next', 'current'];
    const tCount = words.filter(w => temporalMarkers.includes(w)).length;
    const st = Math.min(tCount / Math.max(words.length * 0.3, 1), 1.0);

    const actionVerbs = ['find', 'show', 'compute', 'predict', 'compare', 'analyze',
        'derive', 'calculate', 'identify', 'measure', 'synthesize', 'determine',
        'what', 'how', 'why', 'does', 'is', 'are', 'will', 'can', 'get', 'make',
        'create', 'build', 'run', 'open', 'save', 'search', 'list', 'plot'];
    const aCount = words.filter(w => actionVerbs.includes(w)).length;
    const uniqueActions = new Set(words.filter(w => actionVerbs.includes(w))).size;
    const se = Math.min((aCount + uniqueActions) / Math.max(words.length * 0.4, 1), 1.0);

    const address = toTernary(sk, st, se, 9);

    return {
        sk: Math.round(sk * 1000) / 1000,
        st: Math.round(st * 1000) / 1000,
        se: Math.round(se * 1000) / 1000,
        address
    };
}

function toTernary(sk, st, se, depth) {
    let digits = '';
    let ranges = [[0, 1], [0, 1], [0, 1]];
    const vals = [sk, st, se];

    for (let d = 0; d < depth; d++) {
        const dim = d % 3;
        const [lo, hi] = ranges[dim];
        const third = (hi - lo) / 3;
        let trit;
        if (vals[dim] < lo + third) trit = 0;
        else if (vals[dim] < lo + 2 * third) trit = 1;
        else trit = 2;
        digits += trit;
        ranges[dim] = [lo + trit * third, lo + (trit + 1) * third];
    }
    return digits;
}

export function categoricalDistance(addr1, addr2) {
    let shared = 0;
    const len = Math.min(addr1.length, addr2.length);
    for (let i = 0; i < len; i++) {
        if (addr1[i] === addr2[i]) shared++;
        else break;
    }
    return len - shared;
}

const memory = [];

export function storeEntry(text, response, coords) {
    memory.push({
        text,
        response,
        coords,
        timestamp: Date.now()
    });
}

export function findByDescription(query) {
    if (memory.length === 0) return null;
    const qCoords = computeSEntropy(query);
    let best = null;
    let bestDist = Infinity;

    for (const entry of memory) {
        const dist = categoricalDistance(qCoords.address, entry.coords.address);
        const textMatch = entry.text.toLowerCase().includes(query.toLowerCase().split(' ').slice(-1)[0]);
        const score = dist - (textMatch ? 3 : 0);
        if (score < bestDist) {
            bestDist = score;
            best = entry;
        }
    }
    return best;
}

export function getMemoryCount() {
    return memory.length;
}
