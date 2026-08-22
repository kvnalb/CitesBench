\textbf{CitesBench}

            For decades, peer review has been an integral part of the academic community. It allows scientific institutions such as journals, conferences and grant funding agencies to maintain quality, provide feedback, and allocate scarce attention and resources. However, that mechanism is now under load. The introduction and continuous improvement of Large Language Models(LLMs) has caused  submission volume to explode, even reaching 6 times as much as in the pre-LLM era at some venues. The supply of qualified reviewers has not increased proportionally, and the repeated resubmission of substandard manuscripts adds further to the burden (Zhang et al., 2025). Kuznetsov et al. (2024) describe reviewing as hard, time-consuming, and prone to error.
        
            One solution that the community has considered is the use of LLMs as a complement or substitute to peer review. Studies have shown that the volume of reviewer comments that have signs of AI-use have been increasing over time. Recently, these systems have moved from being an object of study to an intervention deployed at scale, with conferences like the AI-assisted review experiment at NeurIPS 2026 and the supplementary peer review experiment at AAAI-26. Thus, the question of how these LLM-assisted or automated reviews fare in comparison to the human review process has strong operational relevance.

            A growing body of work compares the performance of LLM-based review systems to human reviewers in terms of similarity of scores and comments. While this strand of literature is informative, it might be optimizing for a flawed target. Past work has shown that a fully human review process has features that deviate from selecting the most impactful scientific work. Cortes and Lawrence (2021) determined that 50\% of the variation in reviewer quality scores at NeurIPS 2014 was subjective in origin; Beygelzimer et al. (2023) find that two independent committees reviewing 10\% of NeurIPS 2021 submissions disagreed on accept or reject for 23\% of papers, that approximately half of the accepted list would change if the process were rerun, and that making a conference more selective increases the arbitrariness of the outcome. Among accepted NeurIPS 2014 papers, Cortes and Lawrence report no correlation between reviewer quality scores and citation impact. Among rejected papers, traced to the venues where they eventually appeared, scores and impact do correlate. Their conclusion is that the process was good for identifying poor papers and poor for identifying good ones. Evidence from grant review is mixed but points in the same direction. Li and Agha (2015) find clear benefits of NIH panel evaluation across more than 130,000 grants, particularly for distinguishing high-impact potential among the most competitive applications, while Fang et al.(2016) re-analyze 102,740 funded grants and report that percentile scores are a poor discriminator of productivity, as measured using publications and citation counts.

            Evaluating AI reviewers on outcomes where imitating human reviewers is not the gold standard yields some interesting results. Kim et al. (2026) had 45 domain scientists rate 2,960 criticisms from human and AI reviews of 82 Nature-family papers, finding that a GPT-5.2 reviewing agent outperformed each paper’s top-rated human reviewer on correctness, significance, and evidential sufficiency, while surfacing 26\% of issues missed by all human reviewers. Liang et al. (2024) report that the overlap between points raised by GPT-4 and by human reviewers on ICLR papers, 39.23\%, is comparable to the overlap between two human reviewers, 35.25\%. Weng et al. (2025) train a reviewer model that is more internally consistent by 26.89\% relative to individual human reviewers. On the tasks these studies measure, current models sit at or above human level.
        
            The literature also surfaces some important failure modes. Ho et al. (2026) evaluate 12 frontier models on 1,099 research proposals reconstructed from ICLR submissions and find a pervasive optimism bias, with models rating low-soundness proposals as sound and aggressive prompting shifting errors from false positives to false negatives rather than removing them. Idahl and Ahmadi (2025) observe that general-purpose models tend toward overly positive assessments. 

            We introduce CitesBench, a novel benchmark that models peer review as a function mapping a set of submissions to an admitted subset of fixed size n, scored by an outcome measure over that subset. Our primary sample consists of 4567 papers from ICLR 2018-2020 along with their reviewer scores, sub-scores and area chair decisions. We track citations for both accepted and rejected papers using Semantic Scholar. Because acceptance itself causally raises subsequent citation counts, we estimate that premium at the score cutoff and from rejected papers published elsewhere using a fuzzy regression discontinuity design. We also introduce a suite of tests to quantify memorization of the paper in open model weights using chain of thought monitoring and prompt ablations.  

        This paper makes several contributions to the literature. First, we evaluate reviewing at the level of the admitted set rather than against any per-paper artifact, which tests whether the within-paper competence documented above transfers to a comparative, budget-constrained task. Second, we trace outcome quality across the full range of possible paper budgets in order to surface at what point of the quality curves LLM-reviewers are most helpful. Third, we report results across several different citation based measures (median, log transformed, recall @ top k) in order to show how each system fares across different objectives.


%% results

% comparing LLM regime vs human AC regime -> on citation based metrics
% finding 1: naive LLM based regimes can outperform human regimes on certain metrics. -- baseline paper accepted vs rejected + mean/median citations

%% finding 2: which particular metrics are talking abiut -- average vs outlier outcome

%% findng 3: which LLM regime - one short vs. council 

% finding 4: what are characteristics of agreement vs disagreement. 
%% we look

%% heterogeneity --> at the idea level, or at the PI/team level. 
%% also be done inductively - with no hyptheses

% robustness: memorization (effect attenuated). how do we resolve ties? do citations capture impact? RDD - treatment impact. 


%% Saqib Notes
%% Outline

%% 1. Purpose of the institution (peer review). 
%% Peer review exists to do one thing: select, under a hard attention/resource budget, the work most likely to advance the frontier. Verification at the frontier requires expertise; expertise is scarce; distributed expert judgment was the workable solution.
%% Among its several functions, peer review is consequently an allocation system. It maps a pool of competing submissions into a smaller admitted set under a resource constraint. The quality of the institution depends not only on whether individual reviews are reasonable, but on whether the process identifies the work most likely to advance the scientific frontier.


%% 2. The system is breaking (AI submissions), and LLMs are already being used (formally or informally)
%% Limited human reviewers. But submission explosion (x times more); AI-tinged reviews rising; formal experiments at NeurIPS 2026 / AAAI-26. Deployment is outpacing evaluation.

%% 3. Evidence that LLMs can do good, but are we measuring the right thing? -- human emulation vs. future impact
%% Existing work scores LLM reviewers by agreement with humans (score similarity, comment overlap). But the human process is a noisy instrument for the institution's actual purpose: 50% subjective variance (Cortes & Lawrence), 23% committee disagreement and rerun-instability (Beygelzimer), no score–citation correlation among accepts, mixed grant evidence. So "matches humans" is neither necessary nor sufficient for "selects impactful work."


%% 4. review is a selection function and can we select for impact?
%% introduce the RDD idea. No work on selecting for impact 

%% 5. Why this measurement is hard — and what we do about it? 
%% First, citation outcomes are contaminated by the decision itself: acceptance at a prominent venue causally raises subsequent citations, so naively scoring regimes against realized citations rewards agreement with the historical decision. Second, evaluating historical submissions with contemporary models risks memorized hindsight: a model trained in 2024–2026 may simply recall which 2018–2020 papers became famous (lookahead bias)

%% 6. The pipleine + RDD design 
%% OpenReview + SemanticScholar 
%% Building on existing LLM Reviwer frameworks. Use personas + a head 
%% Relying on local RDDs 

%% 7. Results 


%% 8. Contributions
%% an outcome-based framework -- first to score these systems against realized outcomes rather than human agreement,
%% New evidence on the predictive validity of human peer review, with a machine counterfactual (econ of science literature) -- An LLM regime that beats — or fails to beat — human ACs on the same pool is the first counterfactual evaluator
%% causal estimate of the ICLR acceptance premium 


----

Fig 1: Methodology - Figure

Fig 2: council vs naive LLM vs AC - on different metrics

Fig 3: heterogeneity/mechanism

Tab 1: Summary stats (whats the sample)

Table2 : regression for Fig 2

Table 3: robustness checks (memorization, some alternate metrics, alternate model?/tech variation), 
Memorization
Unmatched papers
RD treatment effect of venue premium

Tab 4: mechanism/heterogeneity

Appendix
Other models
Change the sample to within threshold. 


