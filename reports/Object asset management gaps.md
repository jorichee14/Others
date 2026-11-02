# Object retrieval and asset management: literature and open gaps

Verdicts rest on abstracts and search summaries; arxiv.org could not be opened from the research session. Read the full papers named here before claiming novelty.

## Current literature

**Open-vocabulary object retrieval in robot maps is mature and active.** ConceptGraphs and HOV-SG build open-vocabulary 3D scene graphs from posed RGB-D images ([ConceptGraphs](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf), [HOV-SG, RSS 2024](https://www.roboticsproceedings.org/rss20/p077.pdf)). Newer systems track masks before fusing them ([TRACKGRAPH](https://arxiv.org/abs/2609.31005)), keep small objects as separate instances and decide when a queried object is absent ([OVIP-SG](https://arxiv.org/abs/2608.17633)), and couple memory with search actions ([Searching in Space and Time](https://arxiv.org/pdf/2511.14004)). Objects are stored as point clouds, masks and embeddings, not as 3D assets.

**Generated object assets for simulation and manipulation exist, as one-shot pipelines.** SceneComplete turns one RGB-D view into object meshes with foundation models and raises pick success in clutter from 36.7 % to 73.3 % ([SceneComplete](https://lacuna.tiptreesystems.com/work/scenecomplete-open-world-3d-scene-completion-in-cluttered)). DreamUp3D completes occluded objects from a single RGB-D image ([arXiv 2402.16308](https://arxiv.org/pdf/2402.16308)). SimRecon (CVPR 2026) builds simulation-ready scenes from video by generating each object as an asset, scored with Chamfer distance on 20 ScanNet scenes ([arXiv 2603.02133](https://arxiv.org/pdf/2603.02133)). SAM 3D has been used to place generated objects in Isaac Sim in an industry blog, not yet in a peer-reviewed robotics evaluation ([Marvik](https://www.marvik.ai/blog/generating-3d-assets-for-digital-twin-robotics-environments)).

**Object maps that persist across sessions are a 2026 research front.** OASIS-Map matches objects across revisits and detects appeared, disappeared and moved objects ([arXiv 2607.14899](https://arxiv.org/abs/2607.14899)). LT-Mem (IROS 2026) keeps object identity and event histories over time ([arXiv 2608.19059](https://arxiv.org/abs/2608.19059)). PBD-AG stores object changes as events against a stable baseline ([arXiv 2608.10449](https://arxiv.org/pdf/2608.10449)). REACT keeps updatable object nodes with multi-view instance memory ([arXiv 2503.03412](https://arxiv.org/pdf/2503.03412)). None of them keeps a 3D asset per object.

## Open gaps

| Gap | Closest work | Why it is still open |
|---|---|---|
| A persistent asset registry: one generated 3D asset per object, kept with the object map and updated when objects move or change | SimRecon, SceneComplete (assets, one-shot); OASIS-Map, LT-Mem, REACT (persistent objects, no assets) | Asset generation is built for one-time scene assembly; lifelong object maps are built for identity and retrieval. The two have not been joined. |
| Generated assets checked against the robot's own metric measurements, with each asset's measured and generated parts labelled | SimRecon scores Chamfer distance against reference meshes; SceneComplete reports grasp success | Evaluation compares with a ground-truth mesh or a task outcome, never live against the robot's LiDAR or depth, so nothing tells a planner which parts of an asset to trust. |

## What this means for the planned pipeline

SAM 3 masks with measured point clouds remain the right source for where an object is and how to grasp it. SAM 3D assets, generated on the server at metric scale from the measured point map, are the contribution: kept in a registry with the object map, updated across sessions, and each one checked against the measured cloud with its generated back side labelled. That combination was not found in the literature.
