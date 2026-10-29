# Platform use and feedback

## What ran where

- **Simulation compute:** Brev-provisioned Nebius L40S compute. The public phase-v10 protocol registers provider `nebius`, type `gpu-l40s-d.1gpu-16vcpu-96gb`, and the H resource. The accepted fresh-render result records two 625-frame, 50fps captures and their archive hashes. This is simulation/capture evidence, not a new training or repair result.
- **Model inference:** NVIDIA hosted Nemotron. The completed R1 run used `nvidia/nemotron-3-super-120b-a12b`; its public model record includes the two accepted diagnosis/correction responses and token accounting. Newer Nemotron3.5 investigation evidence remains a separate version and scope.
- **Token Factory:** no runtime call in the evidence supplied for this package. Two promotional $25 issuance notices were confirmed privately. Redemption, compatibility of the two codes, and account balance were not verified. A Builder welcome notice refers to a separate issuance notice and is not an additional credit allocation. Codes and personal account details are excluded.

The compute path above is presented as evidence under the published platform requirements; no individual eligibility approval by the organizers is claimed. NVIDIA's official Brev quickstart uses `nebius.l40sx1.pcie` in its provisioning example; the actual registered type above is a separate identifier: https://docs.nvidia.com/brev/latest/getting-started/quickstart . Rules and FAQ: https://nebiusglobalaihackathon.devpost.com/rules and https://nebiusglobalaihackathon.devpost.com/details/faqs .

## Feedback based on this work

**Nebius compute through Brev:** the provider path supported the Isaac Lab rendering workflow and collection of synchronized captures. We would welcome an exportable receipt joining provider identity, machine type, lifecycle, and artifact hashes, so developers can document platform use without publishing private resource paths. This is compute feedback, not a benchmark of Token Factory or a claim about another project's service quality.

**Token Factory:** the issued-credit and onboarding documentation made the intended inference route identifiable, but we have not run an inference request. We therefore cannot report measured latency, reliability, cost, or model quality on Token Factory. A clear model-ID/structured-output capability table and a redemption receipt would help register a reproducible provider configuration.

**NVIDIA tools and models:** Isaac Lab enables controlled normal/fault comparisons with fixed command evaluations. Nemotron produced accepted structured decisions in the completed R1 and residual-diagnosis records, but its first R1 hypothesis was reward rather than optimizer. Runtime update-count and gradient instrumentation, explicit correction scope, and behavioral validation provided independent checks. A renderer-state synchronization check was also necessary before treating simulation pixels as evidence.
