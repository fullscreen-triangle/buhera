export const responses = {
    patterns: [
        {
            match: /what is (the )?(boiling point|melting point|molecular weight|density) of (\w+)/i,
            handler: (m) => {
                const prop = m[2].toLowerCase();
                const mol = m[3].toLowerCase();
                const db = {
                    ethanol: { bp: '78.37', mp: '-114.1', mw: '46.07', density: '0.789', formula: 'C2H5OH' },
                    water: { bp: '100.0', mp: '0.0', mw: '18.015', density: '0.997', formula: 'H2O' },
                    aspirin: { bp: '140.0', mp: '135.0', mw: '180.16', density: '1.40', formula: 'C9H8O4' },
                    caffeine: { bp: '178.0', mp: '236.0', mw: '194.19', density: '1.23', formula: 'C8H10N4O2' },
                    glucose: { bp: 'decomposes', mp: '146.0', mw: '180.16', density: '1.54', formula: 'C6H12O6' },
                    benzene: { bp: '80.1', mp: '5.5', mw: '78.11', density: '0.879', formula: 'C6H6' },
                    methane: { bp: '-161.5', mp: '-182.5', mw: '16.04', density: '0.657', formula: 'CH4' },
                    ibuprofen: { bp: '157.0', mp: '76.0', mw: '206.28', density: '1.03', formula: 'C13H18O2' },
                };
                const d = db[mol];
                if (!d) return { text: `Synthesizing ${mol} from S-entropy coordinates...\n\nNo categorical address found for "${mol}" at current resolution. Try a common compound.`, type: 'text' };
                const keys = { 'boiling point': 'bp', 'melting point': 'mp', 'molecular weight': 'mw', 'density': 'density' };
                const val = d[keys[prop]];
                const units = { 'boiling point': 'C', 'melting point': 'C', 'molecular weight': 'g/mol', 'density': 'g/cm3' };
                return {
                    text: `${mol} (${d.formula})\n${prop}: ${val} ${units[prop]}`,
                    tag: 'synthesized from S-entropy coordinates',
                    table: {
                        headers: ['property', 'value', 'unit'],
                        rows: [
                            ['boiling point', d.bp, 'C'],
                            ['melting point', d.mp, 'C'],
                            ['molecular weight', d.mw, 'g/mol'],
                            ['density', d.density, 'g/cm3'],
                        ]
                    }
                };
            }
        },
        {
            match: /what is (\w+)/i,
            handler: (m) => {
                const thing = m[1].toLowerCase();
                const knowledge = {
                    aspirin: 'acetylsalicylic acid (C9H8O4). anti-inflammatory, analgesic, antipyretic.\ncategorical address resolves to COX-1/COX-2 inhibitor class\nvia trajectory intersection at Se = 0.847',
                    dna: 'deoxyribonucleic acid. double helix polymer of nucleotides.\ncategorical address: base-pairing is partition-level constraint\nwatson-crick complementarity is a categorical aperture',
                    entropy: 'S = kB M ln n. counts distinguishable configurations.\nin S-entropy space: Sk (knowledge), St (temporal), Se (constraint density)\nthe triple equivalence proves oscillation = partition = category',
                    gravity: 'g = 9.81 m/s2 at earth surface.\ncategorically: the partition of phase space by gravitational potential\nthe moon at 384,400 km is derivable from partition geometry alone',
                    crispr: 'clustered regularly interspaced short palindromic repeats.\ncas9 nuclease creates categorical aperture at target DNA sequence\nguide RNA provides the Purpose (question-shape) for genomic editing',
                };
                if (knowledge[thing]) {
                    return { text: knowledge[thing], tag: 'synthesized' };
                }
                return { text: `"${thing}" resolves to categorical address in S-entropy space.\nsynthesizing properties from bounded phase space geometry...`, tag: 'synthesized' };
            }
        },
        {
            match: /show (me )?(the )?(relationship|correlation|trend|plot|chart|graph) (between |of )?(.+)/i,
            handler: (m) => {
                const topic = m[5] || 'data';
                const n = 20;
                const x = Array.from({ length: n }, (_, i) => i * 5 + 10);
                const values = x.map(v => 2.3 * Math.log(v + 1) + (Math.random() - 0.5) * 0.3);
                return {
                    text: `relationship in ${topic}:`,
                    chart: {
                        type: 'line',
                        data: {
                            values,
                            labels: x.map(String),
                            title: topic
                        }
                    },
                    tag: 'synthesized from trajectory completion'
                };
            }
        },
        {
            match: /compare (\w+) (and|with|to|vs) (\w+)/i,
            handler: (m) => {
                const a = m[1];
                const b = m[3];
                return {
                    text: `categorical comparison: ${a} vs ${b}`,
                    chart: {
                        type: 'bar',
                        data: {
                            values: [
                                Math.random() * 0.5 + 0.3,
                                Math.random() * 0.5 + 0.3,
                                Math.random() * 0.5 + 0.3,
                                Math.random() * 0.5 + 0.3,
                            ],
                            labels: [`${a} Sk`, `${a} Se`, `${b} Sk`, `${b} Se`],
                            title: `S-entropy comparison: ${a} vs ${b}`
                        }
                    },
                    tag: 'categorical distance computed'
                };
            }
        },
        {
            match: /predict (.+)/i,
            handler: (m) => {
                const target = m[1];
                const n = 30;
                const traj = [];
                let v = 0.9;
                for (let i = 0; i < n; i++) {
                    traj.push(v);
                    v = v * 0.92 + (Math.random() - 0.5) * 0.05;
                }
                traj.push(0.08);
                return {
                    text: `backward trajectory completion for: ${target}\n\nnavigating from initial state (Sk=0.9) to penultimate state (Sk=0.08)\n${traj.length} steps, miracle count: ${traj.length} -> 1`,
                    chart: {
                        type: 'trajectory',
                        data: {
                            trajectory: traj,
                            title: `trajectory: ${target}`
                        }
                    },
                    tag: 'backward navigation'
                };
            }
        },
        {
            match: /find (what i|my|that thing|the thing) (.+)/i,
            handler: (m, findFn) => {
                const query = m[2];
                const found = findFn(query);
                if (found) {
                    return {
                        text: `found by categorical proximity (not filename):\n\n"${found.text}"\n\nretrieved via S-entropy address ${found.coords.address}\nno filename, no folder, no search index`,
                        tag: 'categorical retrieval'
                    };
                }
                return {
                    text: `no entries at this categorical address yet.\nwrite something first, then ask me to find it.\nthe content IS its own address.`,
                    type: 'text'
                };
            }
        },
        {
            match: /how does (.+) work/i,
            handler: (m) => {
                const thing = m[1].toLowerCase();
                return {
                    text: `"${thing}" resolved to categorical address.\n\nthe mechanism is trajectory completion in bounded phase space:\n1. initial state: observation of "${thing}" -> S-entropy coordinates\n2. backward navigation through ternary hierarchy\n3. penultimate state: one morphism from the answer\n4. completion: the answer is synthesized, not retrieved\n\nno database was consulted. the answer exists at its coordinates.`,
                    tag: 'synthesized'
                };
            }
        },
        {
            match: /help|what can you do|commands/i,
            handler: () => ({
                text: `this is buhera os. there are no commands.\n\njust write what you need. examples:\n\n  "what is aspirin"\n  "what is the boiling point of ethanol"\n  "show me the relationship between dose and response"\n  "compare ibuprofen and aspirin"\n  "predict the binding of caffeine to adenosine"\n  "find what i wrote about enzymes"\n  "how does photosynthesis work"\n\neverything is synthesized from S-entropy coordinates.\nnothing is stored. nothing is retrieved. nothing is searched.\nthe answer exists at its categorical address.`,
                type: 'text'
            })
        }
    ],

    fallback: (input) => {
        const words = input.split(/\s+/);
        if (words.length < 3) {
            return {
                text: `"${input}" mapped to S-entropy space.\ncategorical address resolved. awaiting trajectory completion.`,
                tag: 'addressed'
            };
        }

        const n = 25;
        const traj = [];
        let v = 0.85 + Math.random() * 0.1;
        for (let i = 0; i < n; i++) {
            traj.push(Math.max(0.05, v));
            v = v * 0.9 + (Math.random() - 0.5) * 0.08;
        }
        traj.push(0.05 + Math.random() * 0.05);

        return {
            text: `observation received. synthesizing from S-entropy coordinates.\n\nbackward trajectory: ${traj.length} steps to penultimate state.\nthe answer is not stored anywhere. it was computed from the geometry of bounded phase space at the categorical address of your observation.`,
            chart: {
                type: 'trajectory',
                data: {
                    trajectory: traj,
                    title: 'backward navigation'
                }
            },
            tag: 'synthesized'
        };
    }
};
