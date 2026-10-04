You're right to be cautious here. What you saw is promising, but on its own it doesn't prove the "one network, several strategies" claim.

**Does it prove the shared network stores different strategies?** Not yet. The policy you watched spent its last 300M steps trained on that one body only. From about 800M onward it only saw one region of the design space. It could simply have specialised, and the random-body eval (cost 219) shows it no longer handles other bodies well. Two cheap checks on the existing run's checkpoints would settle it:
- **Same weights, different bodies.** Take a checkpoint from before the first pruning (around 350M, when all 8 Gaussians were alive). Run it on a long-legged body and on a short-legged one, and measure route choice, e.g. how much of each episode the torso spends over a hazard. If long legs step over and short legs weave under identical weights, that's your evidence.
- **When the strategy appeared.** Run the evolved body under the 400M, 800M, 1.2B, 1.7B and 2B checkpoints with the same metric. The long-legged Gaussian must already have been faster by 400–800M to win the prunings, but that could have come from longer strides between hazards rather than stepping over them. This sweep answers whether the final fine-tune is what switched the strategy.

You'd need to copy those checkpoints over (about 10 MB each); then I can run both.

**Has it been shown before?** One network producing different gaits for different bodies is established in the universal-controller work (MetaMorph, Amorpheus). As far as I remember, Schaff et al. mostly report designs and returns rather than analysing strategies. What looks new is the safety angle: a constraint changing both the body and the route. A quick search found co-design with resource limits on the design, and constrained RL for legged locomotion, but not safe RL inside the co-design loop. That's promising but not a full literature review, so do a proper one before claiming novelty.

**On "bigger = better":** the result is really "longer legs, smaller and lighter torso". The actual weakness is that 4 of 7 parameters are pinned at the edge of the allowed range, so the range limit, not a trade-off, decided the answer. A reviewer would spot that immediately, and there are two causes:
- **Size is free.** Motor strength is rescaled with leg length (gear 191 vs 150), so longer legs come with extra torque at no cost.
- **Both pressures point the same way.** In this environment, long legs are both faster and better at stepping over mines, so safety and speed never compete.

**What I'd suggest, in order:**

1. **An environment where safety pulls the body the other way.** Your lasers idea is the strongest option: beams at random heights and positions, combined with the mines. Long legs step over mines but hit beams; low bodies duck beams but must weave between mines. The best body then depends on the mix and on the budget, which is exactly the tension you want. The Lasers task already exists, so the main work is randomising the beam heights.
2. **A budget sweep as the headline figure.** Co-design at several budgets (no constraint, 25, 5, 2, 0) and plot the evolved design parameters against the budget, showing that safety requirements reshape the body. Each run is about 15 h.
3. **The controls any paper needs:**
   - unconstrained co-design (`PENALIZER=none`) on the same task: does it pick the same body?
   - evolve without safety first, then train a safe policy on that body
   - the nominal body with safe RL
   - shared vs per-Gaussian λ (your method contribution)
   - 3 seeds for the main conditions
4. **Make size cost something.** Either keep motor strength fixed, add an energy term, or give the design a fixed mass budget. Otherwise every environment tends to push toward the range edge.
5. **Payload (your backpack idea)** as a second tension: a heavy or fragile load with a cost on tipping or jerk. That pushes toward a wide, low, stable stance, against long legs.
6. **Real-world framing without a new robot.** The lattice minefield already looks like a crop field. Legged robots that mustn't trample plants (mines) or hit low branches (lasers) is a real safety problem, and a nuclear or rescue site works too. A real quadruped (e.g. Menagerie's Go1) would be more convincing, but reshaping mesh-based robot models is a lot of engineering. I'd keep the ant for the main results and add a second robot later if a reviewer asks.

Boxes would also stop "big" from being free, but they only add a physical limit, not a safety trade-off, so I'd prioritise lasers.

If you agree, I'd start with the two cheap checks on this run plus an unconstrained co-design baseline, and build the randomised lasers + mines environment in parallel.

Sources:
- [Co-design is powerful and not free](https://arxiv.org/html/2510.08368)
- [Co-design of Embodied Neural Intelligence via Constrained Evolution](https://arxiv.org/pdf/2205.10688)
- [Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion](https://arxiv.org/html/2308.12517v2)
- [N-LIMB: Neural Limb Optimization for Efficient Morphological Design](https://arxiv.org/pdf/2207.11773)