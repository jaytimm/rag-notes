# RAG systems in R

Notes and thoughts.

-   [RAG systems in R](#rag-systems-in-r)
    -   [Query expansion](#query-expansion)
        -   [Augmented query](#augmented-query)
        -   [Step-back prompting](#step-back-prompting)
        -   [Hypothetical answer (HyDE)](#hypothetical-answer-(hyde))
        -   [Hypothetical questions](#hypothetical-questions)
    -   [Building contexts](#building-contexts)
        -   [Sentence Window Retrieval](#sentence-window-retrieval)
    -   [Evaluation](#evaluation)

## Query expansion

``` r
library(dplyr)
query99 <- 'Why is Generation X wary of President Biden \n
            as the 2024 presidential election looms? \n'

system99 <- 'You are a political analyst.'
```

> A simple utility function for chat completion via Open AI.

``` r
get_chat_response <- function(prompt, system){
  
  resp <- openai::create_chat_completion(
      model = "gpt-4",
      messages = list(list("role" = "system", 
                           "content" = system),
                      list("role" = "assistant", 
                           "content" = prompt)) 
      )
  
  # Extract generated text
  resp0 <- resp$choices$message.content
  # Re-format/clean
  resp1 <- Filter(function(x) x != "", strsplit(resp0, '\n')[[1]])
  data.frame(resp = resp1)
}
```

### Augmented query

> These generated queries are designed to explore different facets or
> interpretations of the original query, thereby broadening the scope of
> the search.

1.  Use LLM to generate similar queries to original user query
2.  Embed original and augmented queries
3.  Retrieve documents (via semantic search) using both the original and
    augmented query vectors
4.  Consolidate highest ranked retrieved documents as relevant contexts
    for LLM

``` r
p1 <- 'For the question above, suggest up to five additional related questions.\n
       Keep suggested questions short.\n
       Suggest a variety of questions that cover different aspects of the topic.\n 
       Output one question per line. \n
       Do not number the questions.'

get_chat_response(prompt = paste0(query99 |> toupper(), p1),
                  system = system99) |> 
  knitr::kable()
```

| resp                                                                                         |
|:-----------------------------------------------------------------------|
| What are the major concerns of Generation X about President Biden’s policies?                |
| How did Generation X vote in past presidential elections, specifically in 2020’s election?   |
| What could President Biden do to earn the trust and confidence of Generation X?              |
| Has the Biden administration addressed the issues specifically centered around Generation X? |
| How does the political ideology of Generation X differ from other generations?               |

### Step-back prompting

1.  Use LLM to provide a more general user query
2.  Embed original and more general queries
3.  Retrieve documents using both the original and more general vectors
4.  Aggregate highest ranked retrieved documents

``` r
p2 <- 'Provide five *more general* versions of the question above. \n
       Output one question per line. Do not number the questions.'

get_chat_response(prompt = paste0(query99 |> toupper(), p2),
                  system = system99) |> 
  knitr::kable()
```

| resp                                                                                                                                 |
|:-----------------------------------------------------------------------|
| What causes Generation X’s apprehensions towards President Biden as the 2024 elections approach?                                     |
| Why is there a sense of unease among Generation X towards President Biden in the context of the upcoming 2024 Presidential Election? |
| Can we delve into the reasons behind Generation X’s reservations about President Biden’s prospects in the 2024 election?             |
| Why are the individuals from Generation X skeptical about President Biden’s campaign for the 2024 Presidential elections?            |
| What are the factors contributing to the wariness of Generation X towards President Biden’s 2024 election campaign?                  |

### Hypothetical answer (HyDE)

1.  Use LLM to provide a hypothetical answer to user query
2.  Embed user query & hypothetical answer
3.  Retrieve documents using both query & hypothetical answer (or a
    concatenation of the two) vectors
4.  Aggregate highest ranked retrieved documents

``` r
hyde <- 'Provide an example answer to the question above. \n
         Limit to 50 words.'

get_chat_response(prompt = paste0(query99 |> toupper(), hyde), 
                  system = system99) |> 
  knitr::kable()
```

| resp                                                                                                                                                                                                                                                                                                                                                      |
|:-----------------------------------------------------------------------|
| Generation X’s wariness towards President Biden is influenced by concerns over economic instability, debates on free speech, and policies perceived as leaned towards younger and older demographics. Particularly, fears about increased taxes, inflation, and contentious social issues might sway their preference for the 2024 Presidential Election. |

### Hypothetical questions

1.  Use LLM to generate specific questions for each text chunk
2.  Embed chunk questions
3.  Retrieve documents via question vectors
4.  Aggregate highest ranked retrieved documents

``` r
library(dplyr)
articles <- textpress::web_scrape_urls(x = 'Generation X', 
                                       cores = 6)

chunks <- articles |>
  filter(!is.na(text)) |>
  mutate(doc_id = row_number()) |>
  
  textpress::nlp_split_sentences() |>
  textpress::rag_chunk_sentences(chunk_size = 2, 
                                 context_size = 1)
```

``` r
n = 3
set.seed(999)
chunks_sample <- chunks |>
  filter(grepl('Generation X|Gen X', chunk_plus_context)) |>
  sample_n(n)

answers <- paste0(chunks_sample$chunk_plus_context,  collapse = '\n\n')

question <- 'For each of the contexts provided below, \n
             generate a hypothetical question for which \n
             the context could serve as an answer.\n
             Output one question per line. \n
             Keep questions simple. Avoid "compound" questions. \n
             Do not number the questions.'

hyp_question <- get_chat_response(prompt = paste0(question, answers),
                                  system = '')

hyp_question[1:n,] |> 
  cbind(chunks_sample$chunk_plus_context) |>
  knitr::kable()
```

|                                                                                                         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|:-----------|:-----------------------------------------------------------|
| What does being a part of Generation X mean to Pinnock in her current role?                             | Pinnock says being Gen X serves her well in her current role. <b>She says she and her friends call their generation the MacGyvers, after the 1980s television character who could figure his way out of any jam using his own wits and knowledge. “We were the only generation to know what it was like to not work with technology but are not scared to work with it,” Pinnock says.</b> “You had to rely on yourself.                                                                                                                                                                                            |
| What factors have caused a discrepancy between the mean and median retirement savings for Generation X? | The reason for the vast gulf between the mean and the median is the 40% of the demographic that has absolutely nothing saved for retirement, which drags down the average like an anchor. <b>Gen X was the first generation to enter the labor force after employers transitioned to defined contribution accounts like 401(k) plans. “While the generation before them could rely on pension plans, most Gen Xers have had to depend on themselves for retirement savings,” said Laura Sterling of Georgia’s Own Credit Union, the second-largest credit union in Georgia.</b> “Pensions are a thing of the past.” |
| At what age did Generation X workers typically begin saving for retirement according to the study?      | Consequentially, Generation X fell behind on their savings before they even knew it was time to get started.” <b>According to the research, which surveyed 5,725 employees, the median age when Gen X workers began saving for retirement was 30, as compared with Millennials (25), Gen Z (19). Baby Boomers, who were working in the age of pensions and very early 401(k) plans, started at a median age of 35.</b> Now only 17% of Gen Xers feel very confident they will be able to retire with a comfortable lifestyle, according to the study conducted in late 2022.                                        |

## Building contexts

### Sentence Window Retrieval

## Evaluation
