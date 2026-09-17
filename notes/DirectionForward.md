1. MPC — no; a foothold planner + tracking policy — this is what I'd actually do. The plan's objection to MPC over the physics still stands (tens of millions of physics steps per morphology evaluated, and it doesn't discover gaits). But the hierarchical version is a different thing: a geometric planner picks safe footholds — we already have that code, the kinematic check is exactly a foothold planner and it found the zero-cost path — and the policy gets the next target foothold per foot in its observation with a dense tracking reward. That turns "discover placement from a cost" into "hit a target", which RL does well. Safety becomes a property of the planner; morphology enters as reachability and tracking accuracy, which is a clean co-design question. It's also what works in the literature: the hierarchical Doggo result was 79% success where PPO-Lagrangian was 0. Moderate build (jittable planner in the env, target obs, tracking term).

2. What the perceptive-locomotion literature does for stepping-stone-type tasks is give the policy an egocentric terrain scan: a grid of terrain samples around the robot in its yaw frame (Miki et al. 2022, the ANYmal stepping-stone work, most "learning to walk on discrete footholds" papers). Our analogue is trivial to compute: a grid of cells in the torso's yaw frame, 1 where the cell centre is inside a hazard disc. Continuous, no nearest-neighbour switching, and it shows the free space directly; the network already knows where its feet are in that frame from the joint angles. Three versions, by cost:

representation	dims	what it gives
hazard grid around the torso, 8×8 at 0.15 m	64	the field-standard scan; free space, not just obstacles
per-foot grid, 5×5 at 0.08 m around each foot's ground projection	100	the same map, centred where the decision is made
touchdown clearance per foot: clearance at the ballistic landing point (foot xy + v·(z/vz) while descending)	+4	patches weakness 2 into the existing obs, cheap
My recommendation: the torso hazard grid, plus touchdown clearance on top of the existing 16 — the grid is general (any layout, any body), the touchdown feature is the one number placement actually turns on. ~60 lines in foot_obstacle_observations' neighbourhood, a new --hazard_grid flag, and C/B become "grid" arms instead of "nearest-mine" arms — worth doing before C goes out, since B inherits C's obs width. Say the word and I'll build it.

3. Gait-parameterised actions (PMTG/CPG-style: the policy outputs stride length, phase, clearance over a trajectory generator instead of raw torques). Foot placement becomes a low-dimensional, semantic decision rather than something emergent from 8 torques at 12 Hz. Standard in legged robotics; the biggest build on this list.

4. Imitation from the planner. The kinematic check produces a zero-cost foothold sequence; use it as a reference for a tracking reward or DAgger, DeepMimic-style. Sits between the hierarchy I described and pure RL: the planner is used at training time only, the deployed policy is flat.

5. Read all three against the actual code. The pasted advice is mostly right, one claim is off, and the sources disagree with two of my choices.

What the sources actually do

hazard cost	where	grounding gate	what the policy sees
Safety Gym / Gymnasium (hazards.py:85-89)	cost × (size − dist) — linear in penetration depth, torso centre	cost	none (2D)	torso lidar
CRAX Pathway (safe_pathway.py:412-473)	(1 − d/r)² per foot, quadratic, max over feet, sum over hazards, ×1.5; plus 5.0 cost on unhealthy termination	cost	foot z < 0.10 m	torso lidar only (planar robots — it's a timing task, not 2D placement)
Hwang et al. 2026	none — stepping stones, falling is the failure	—	—	17×11 heightmap at 0.1 m + per-foot Gaussian blobs (w=10, σ=0.1) on the same grid, CNN + attention; the foot map alone was +10% success, the biggest component
So "the field has converged on penetration depth" is right — even the original Safety Gym hazard cost we ported is graded, and ours went binary somewhere in the port. "Quadratic" is CRAX; Safety Gym is linear. The CaT "13×11 at 8 cm" detail I can't verify locally — treat as unconfirmed.

Where we differ, and what I'd change

The graded term belongs in the cost. Both references make penetration the cost; I put it in the reward. Your instinct was right and the sources back it. Fix: cost = Σ_hazards max_feet(depth) (linear like Safety Gym, quadratic as an option), with the binary count still logged as episode_hazard_steps so the new numbers stay readable. Reward shaping stays available for the unconstrained arms only.
No ring outside the disc. Neither reference ramps beyond the hazard edge; my 0.25 margin is nonstandard. Default the ramp to the disc radius.
Charge the flip. CRAX's 5.0 termination cost is the answer to the exit both CRPO and Lagrangian took: if ending the episode costs about a careless crossing's worth, "walk 3 m and fall over" no longer satisfies the budget. A --flip_cost flag, off by default.
Foot-resolution obs. Hwang's point is feet and terrain in one representation so the network can relate them; CRAX doesn't do it at all. The MLP-scale version is the per-foot local hazard grid I suggested — the grid is the source's idea, the per-foot centring is our substitute for their attention encoder.
One thing the sources are looser on: grounding. CRAX charges a foot hovering up to 10 cm; Safety Gym charges the torso passing over. Our 1 mm is the physically honest choice you asked for, but it's the strictest in the literature and the signal is correspondingly sparser — worth remembering if the graded cost still doesn't bite.

The velocity-tracking warning doesn't apply: our reward is global +x with walls and a lattice spanning the corridor, so the field can't be dodged — the analogous exploit here is stopping/flipping, which is what item 3 closes.

Items 1–3 are ~80 lines and change the budget's units (penetration units, ~0.8× hazard-steps on the walker), so the constrained arm's budget would be re-derived. Want me to build them before the cluster arms go out?

6. Switch to CaT (constraints as terminations). This is the closest published method to what you're doing and it's about five lines on top of PPO. Instead of a hard budget, you compute a termination probability δ proportional to the magnitude of the constraint violation, normalized by an exponential moving average of the max violation in the recent batch, then multiply rewards by (1−δ) and write δ into the dones. Allowing δ strictly between 0 and 1 is what lets the agent learn to recover from violations and explore a little inside the violating region. They anneal soft constraints from p_max 0.05 up to 0.25 over training while keeping genuine no-go constraints at 1.0. They ran 60+ constraint terms this way on real hardware.

If you want to stay closer to safe-RL-proper: PPO-Lagrangian with the PID multiplier update (Stooke et al.) is the standard strong baseline. Skip CPO — Ray et al. found it performs surprisingly poorly on Safety Gym relative to Lagrangian methods. In the legged world, Kim et al. use IPO with adaptive constraint thresholding and note that CPO's optimization cost grows linearly in the number of constraints. 
OpenAI
arxiv