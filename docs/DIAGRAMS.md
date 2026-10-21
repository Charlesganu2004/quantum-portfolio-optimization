# Diagrams

Each diagram below is also a standalone Mermaid file in `docs/diagrams/`, which
`scripts/render_diagrams.ps1` renders to PNG. This page holds the same text as
fenced `mermaid` blocks so GitHub draws them in place.

## Architecture

The modules and what each one is allowed to touch. Source: `docs/diagrams/architecture.mmd`.

```mermaid
---
config:
  theme: base
  themeVariables:
    fontFamily: "Arial, Helvetica, sans-serif"
    fontSize: "15px"
  flowchart:
    curve: basis
    nodeSpacing: 40
    rankSpacing: 50
---
flowchart TB
    subgraph L1["DATA LAYER"]
        A1["Alpaca IEX bars<br/><i>real, needs paper keys</i>"]
        A2["Synthetic generator<br/><i>3-factor, no credentials</i>"]
        A3["Features<br/>annualized mu, Ledoit-Wolf Sigma"]
        A1 --> A3
        A2 --> A3
    end

    subgraph L2["OPTIMIZATION LAYER: same inputs, same objective"]
        FAN(("same<br/>inputs"))
        B1["Mean-variance<br/>relax then truncate"]
        B2["Heuristic<br/>backward elimination + swap"]
        B3["QUBO annealing<br/>quantum-inspired"]
        B4["QAOA<br/>simulated, 14 qubit cap"]
        JOIN(("candidate<br/>weights"))
        FAN --> B1 --> JOIN
        FAN --> B2 --> JOIN
        FAN --> B3 --> JOIN
        FAN --> B4 --> JOIN
    end

    subgraph L3["REVIEW LAYER: advisory only, cannot authorize"]
        C1["Knowledge base<br/>provenance records"]
        C2["local model (llama3.2:3b)<br/>or offline rules stand-in<br/>schema-constrained"]
        C3["Citation verifier<br/>deterministic"]
        C1 --> C2 --> C3
    end

    subgraph L4["CONTROL LAYER: gate approves or rejects each entry"]
        D1["Risk gate<br/>18 reject reasons<br/>fixed-size entry, gate derives the stop"]
        D2["Paper broker<br/>simulated fills<br/>exit sells skip the gate"]
        D3["Live broker<br/>DISABLED, raises on call"]
        D1 --> D2
        D1 -. "no code path" .-> D3
    end

    A3 --> FAN
    JOIN -- "candidate weights &nbsp;" --> C2
    C3 -- "verdict only &nbsp;" --> D1

    classDef data fill:#E8F0FA,stroke:#1F4E79,stroke-width:1.5px,color:#12263F
    classDef opt fill:#EAF3EC,stroke:#2E7D32,stroke-width:1.5px,color:#12331A
    classDef llm fill:#FFF1DC,stroke:#B36B00,stroke-width:1.5px,color:#4A2E00
    classDef ctl fill:#FDE5E2,stroke:#B03020,stroke-width:2px,color:#4A0F0A
    classDef off fill:#EEEEEE,stroke:#888888,stroke-dasharray: 5 4,color:#555555
    classDef hub fill:#2E7D32,stroke:#1B5E20,color:#FFFFFF

    class A1,A2,A3 data
    class B1,B2,B3,B4 opt
    class C1,C2,C3 llm
    class D1,D2 ctl
    class D3 off
    class FAN,JOIN hub

    style L1 fill:#F7FAFF,stroke:#9DB8D6,stroke-dasharray: 5 4
    style L2 fill:#F6FBF7,stroke:#93C39C,stroke-dasharray: 5 4
    style L3 fill:#FFFAF2,stroke:#D9A45B,stroke-dasharray: 5 4
    style L4 fill:#FFF6F5,stroke:#D98B80,stroke-dasharray: 5 4
```

## Decision pipeline

One step of `SessionRunner`, from data to report. Source: `docs/diagrams/pipeline.mmd`.

```mermaid
---
config:
  theme: base
  themeVariables:
    fontFamily: "Arial, Helvetica, sans-serif"
    fontSize: "16px"
  flowchart:
    curve: basis
    nodeSpacing: 40
    rankSpacing: 45
---
flowchart LR
    A["Market data"]
    B["Features: mu, Sigma"]
    C["Optimizer<br/>produces candidate weights"]
    D["Fixed evidence set<br/>4 formula records"]
    E["Reviewer<br/>local model or offline rules"]
    F["Risk gate<br/>18 reject reasons"]
    G["Paper order"]

    A --> B --> C --> D --> E
    E -- "verdict only" --> F
    F -- "approve or reject<br/>a fixed-size entry" --> G

    C -. "the reviewer can only<br/>subtract from this set" .-> F

    N1["Prompt contains no dollar amounts.<br/>Asserted by test."]
    N2["Model output can never<br/>become a number on an order."]
    E -.- N1
    F -.- N2

    classDef det fill:#E8F0FA,stroke:#1F4E79,stroke-width:1.5px,color:#12263F
    classDef llm fill:#FFF1DC,stroke:#B36B00,stroke-width:2px,color:#4A2E00
    classDef auth fill:#FDE5E2,stroke:#B03020,stroke-width:2.5px,color:#4A0F0A
    classDef note fill:#F5F5F5,stroke:#AAAAAA,stroke-dasharray: 3 3,color:#444444
    class A,B,C,D det
    class E llm
    class F,G auth
    class N1,N2 note
```

## Reviewer states

Every failure inside a review ends in ABSTAIN. Source: `docs/diagrams/reviewer_states.mmd`.

```mermaid
---
config:
  theme: base
  themeVariables:
    primaryColor: "#E8F0FA"
    primaryTextColor: "#12263F"
    primaryBorderColor: "#1F4E79"
    lineColor: "#44546A"
    fontFamily: "arial, helvetica, sans-serif"
    fontSize: "15px"
  flowchart:
    curve: basis
    padding: 20
    nodeSpacing: 55
    rankSpacing: 65
---
flowchart LR
    S(["&nbsp;Request built&nbsp;<br/>&nbsp;no dollar amounts&nbsp;"])
    R{"&nbsp;Model&nbsp;<br/>&nbsp;reachable?&nbsp;"}
    P{"&nbsp;Parses to&nbsp;<br/>&nbsp;schema?&nbsp;"}
    E{"&nbsp;APPROVE has&nbsp;<br/>&nbsp;evidence?&nbsp;"}
    C{"&nbsp;Citations&nbsp;<br/>&nbsp;resolve?&nbsp;"}
    V(["&nbsp;Verdict honored&nbsp;<br/>&nbsp;APPROVE / REJECT / ABSTAIN&nbsp;"])
    A(["&nbsp;ABSTAIN&nbsp;<br/>&nbsp;no trade&nbsp;"])

    S --> R
    R -- "yes" --> P
    P -- "yes" --> E
    E -- "yes" --> C
    C -- "yes" --> V

    R -- "no: unreachable&nbsp;" --> A
    P -- "no: schema violation&nbsp;" --> A
    E -- "no: unsupported claim&nbsp;" --> A
    C -- "no: fabricated source&nbsp;" --> A

    N["&nbsp;Within a review, every failure edge terminates at ABSTAIN.&nbsp;<br/>&nbsp;Outside it: with no model reachable at startup,&nbsp;<br/>&nbsp;resolve_reviewer falls back to the offline rules&nbsp;<br/>&nbsp;reviewer, which can approve.&nbsp;"]
    A -.-> N

    classDef ok fill:#E3F5E6,stroke:#2E7D32,stroke-width:2px,color:#12331A
    classDef bad fill:#FDE5E2,stroke:#B03020,stroke-width:2.5px,color:#4A0F0A
    classDef dec fill:#FFF6DC,stroke:#B08900,stroke-width:1.5px,color:#3D2E00
    classDef note fill:#F2F2F2,stroke:#999999,stroke-width:1px,color:#333333
    class V,S ok
    class A bad
    class R,P,E,C dec
    class N note
```

## QUBO method

How the cardinality cap becomes a penalty and how the weights are set. Source: `docs/diagrams/qubo.mmd`.

```mermaid
---
config:
  theme: base
  themeVariables:
    fontFamily: "Arial, Helvetica, sans-serif"
    fontSize: "15px"
---
flowchart LR
    IN["Inputs<br/>mu, Sigma, lambda, K, w_max"]

    IN --> Q1["Build QUBO<br/>E(x) = -(1/K) mu'x + (lambda/K^2) x'Sigma x + P(sum x - K)^2"]
    Q1 --> Q2["Expand using x_i squared = x_i<br/>gives symmetric Q with E(x) = x'Qx"]
    Q2 --> Q3{"Move type?"}

    Q3 -- "single bit flip<br/>(naive)" --> BAD["Cardinality changes<br/>penalty activates and dominates<br/>search degenerates to any feasible subset"]
    Q3 -- "swap: one held for one unheld<br/>(this implementation)" --> GOOD["State never leaves exactly-K set<br/>penalty contributes EXACTLY ZERO to delta E<br/>pure objective landscape is searched"]

    GOOD --> ANN["Metropolis annealing<br/>delta E = 2(h_j - h_i) + Q_ii + Q_jj - 2Q_ij<br/>O(1) to evaluate, O(n) to apply"]
    ANN --> REP["Repair to exact cardinality<br/>feasibility guaranteed, not hoped for"]
    REP --> QP["Continuous weights from the SAME QP<br/>every other method uses"]
    QP --> OUT["Portfolio weights<br/>comparison isolates SELECTION, not weighting"]

    classDef good fill:#E3F5E6,stroke:#2E7D32,stroke-width:2px,color:#12331A
    classDef bad fill:#FDE5E2,stroke:#B03020,stroke-width:2px,color:#4A0F0A
    classDef step fill:#E8F0FA,stroke:#1F4E79,stroke-width:1.5px,color:#12263F
    class GOOD,OUT good
    class BAD bad
    class IN,Q1,Q2,ANN,REP,QP step
```

## Reviewer ablation

The two arms of `compare_review_ablation` and how they are compared. Source: `docs/diagrams/ablation.mmd`.

```mermaid
---
config:
  theme: base
  themeVariables:
    fontFamily: "Arial, Helvetica, sans-serif"
    fontSize: "15px"
---
flowchart LR
    SEED["Identical seed, prices,<br/>optimizer and estimator"]

    SEED --> ARM1
    SEED --> ARM2

    subgraph ARM1["Arm A: review enabled"]
        direction TB
        O1["Optimizer produces candidate"]
        R1["LLM reviewer"]
        G1["Risk gate"]
        T1["0 of 12 steps traded<br/>(both optimizers)"]
        O1 --> R1 --> G1 --> T1
    end

    subgraph ARM2["Arm B: review bypassed"]
        direction TB
        O2["Optimizer produces candidate"]
        B2["Reviewer skipped<br/>recorded as BYPASSED, not APPROVE"]
        G2["Risk gate"]
        T2["QUBO: 12 of 12 steps traded<br/>heuristic: 11 of 12"]
        O2 --> B2 --> G2 --> T2
    end

    T1 --> CMP
    T2 --> CMP

    CMP{"Compare<br/>decision by decision"}
    CMP --> RES

    RES["100 percent of QUBO decisions changed<br/>(heuristic: 91.7 percent)<br/><br/>But: no trade in either arm with review on.<br/>Main-run log: 12 abstentions<br/>(11 verdicts, 1 failed citation check).<br/>Review avoided the heuristic loss of -1.6007 percent<br/>AND forfeited the QUBO gain of +0.9526 percent."]

    VERDICT["Degenerate result.<br/>Uniformly conservative, not selective.<br/>Safety property holds. Decision value not supported."]
    RES --> VERDICT

    classDef arm fill:#E8F0FA,stroke:#1F4E79,stroke-width:1.5px,color:#12263F
    classDef warn fill:#FDE5E2,stroke:#B03020,stroke-width:2.5px,color:#4A0F0A
    classDef neutral fill:#F5F5F5,stroke:#777777,color:#222222
    class O1,R1,G1,O2,B2,G2 arm
    class VERDICT,RES warn
    class SEED,CMP,T1,T2 neutral
    style ARM1 fill:#FFFAF2,stroke:#D9A45B,stroke-dasharray: 4 3
    style ARM2 fill:#F7FAFF,stroke:#9DB8D6,stroke-dasharray: 4 3
```

## Halt latch

The consecutive-loss halt stays engaged for the rest of a session. Source: `docs/diagrams/halt_latch.mmd`.

```mermaid
---
config:
  theme: base
  themeVariables:
    fontFamily: "Arial, Helvetica, sans-serif"
    fontSize: "15px"
---
stateDiagram-v2
    direction LR

    state "Trading permitted<br/>consecutive_losses = 0" as OK
    state "1 consecutive loss" as L1
    state "2 consecutive losses" as L2
    state "HALTED<br/>gate blocks every order" as H
    state "Step scores exactly 0.00<br/>no order was submitted" as Z

    [*] --> OK
    OK --> L1 : losing trade
    L1 --> L2 : losing trade
    L2 --> H : 3rd losing trade
    L1 --> OK : winning trade
    L2 --> OK : winning trade

    H --> Z : gate rejects the entry
    Z --> H : no trade, so the counter cannot change

    note right of H
        Deliberate: the halt stays
        latched for the rest of the
        session. A control that silently
        resumes is not a control.
        halted_at_interval records when
        it engaged, and statistics use
        only the steps before it.
    end note

    note right of Z
        Only traded steps move the
        counter. A blocked step is
        neither a win nor a loss.
        All 30 runs in
        reports/trade_analysis.json
        halted, at a mean step of 12.8
        to 16.8 out of 200.
    end note
```
