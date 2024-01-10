-   [Some notes on retrieval augmented generation (RAG) in
    R](#some-notes-on-retrieval-augmented-generation-rag-in-r)
-   [Query expansion](#query-expansion)
    -   [Augmented query](#augmented-query)
    -   [“Step-back” prompting](#step-back-prompting)
    -   [Hypothetical answer (HyDE)](#hypothetical-answer-hyde)
    -   [Hypothetical questions](#hypothetical-questions)
-   [Building contexts](#building-contexts)
    -   [Sentence Window Retrieval](#sentence-window-retrieval)
-   [Evaluation](#evaluation)

## Some notes on retrieval augmented generation (RAG) in R

-   [Some notes on retrieval augmented generation (RAG) in
    R](#some-notes-on-retrieval-augmented-generation-(rag)-in-r)
-   [Query expansion](#query-expansion)
    -   [Augmented query](#augmented-query)
    -   [Step-back” prompting](#step-back%22-prompting)
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

| resp                                                                                |
|:-----------------------------------------------------------------------|
| What particular policies of President Biden does Generation X disagree with?        |
| How has President Biden’s performance impacted the political views of Generation X? |
| What are the key issues for Generation X in the 2024 Presidential election?         |
| How did Generation X vote in the 2020 election, and why might this change in 2024?  |
| What steps could President Biden take to gain more support from Generation X?       |

### “Step-back” prompting

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

| resp                                                                                                   |
|:-----------------------------------------------------------------------|
| Why are members of Generation X skeptical about President Biden?                                       |
| What are the reasons for Generation X’s apprehension about President Biden as 2024 elections approach? |
| Why might Generation X be hesitant to support President Biden in the 2024 Presidential election?       |
| In what ways has President Biden’s leadership caused concern among Generation X?                       |
| What factors could be leading Generation X to question President Biden’s potential 2024 candidacy?     |

### Hypothetical answer (HyDE)

1.  Use LLM to provide a hypothetical answer to user query
2.  Embed user query & hypothetical answer
3.  Retrieve documents using both query & hypothetical answer (or a
    concatenation of the two) vectors
4.  Aggregate highest ranked retrieved documents

``` r
hyde <- 'Provide an example answer to the question above, limited to 50 words: \n\n'

get_chat_response(prompt = paste0(query99 |> toupper(), hyde), 
                  system = system99) |> 
  knitr::kable()
```

| resp                                                                                                                                                                                                                                                                                                                   |
|:-----------------------------------------------------------------------|
| Generation X, generally conservative in financial matters, is wary of President Biden due to perceived fiscal irresponsibility, increased taxation proposals, and concerns about inflation. Many also express dissatisfaction with his handling of issues like immigration, foreign policy, and the COVID-19 pandemic. |

### Hypothetical questions

1.  Use LLM to generate specific questions for each text chunk
2.  Embed chunk questions
3.  Retrieve documents via question vectors
4.  Aggregate highest ranked retrieved documents

``` r
library(dplyr)
articles <- textpress::web_scrape_urls(x = 'Generation X', cores = 6)

chunks <- articles |>
  filter(!is.na(text)) |>
  mutate(doc_id = row_number()) |>
  
  textpress::nlp_split_sentences() |>
  textpress::rag_chunk_sentences(chunk_size = 2, 
                                 context_size = 1)
```

``` r
set.seed(99)
chunks_sample <- chunks |>
  filter(grepl('Generation X|Gen X', chunk_plus_context)) |>
  sample_n(3)

answers <- paste0(chunks_sample$chunk_plus_context,  collapse = '\n\n')

question <- 'For each of the contexts provided below, \n
             generate a hypothetical question for which \n
             the context could serve as an answer.\n
             Output one question per line. \n
             Keep questions simple. Avoid "compound" questions. \n
             Do not number the questions.'

hyp_question <- get_chat_response(prompt = paste0(question, answers),
                                  system = '')

hyp_question[1:3,] |> 
  cbind(chunks_sample$chunk_plus_context) |>
  knitr::kable()
```

|                                                                                                                                                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|:-------------|:---------------------------------------------------------|
| What was the impact of stimulus money on excess cash and how does it affect retirement saving efforts, especially for Gen X?                       | So there was perhaps more excess cash out there lingering around because of the massive stimulus.” <b>Michael Shamrell, vice president of Fidelity’s workplace investing thought leadership, shares how to best save for retirement based on your age and salary. The report also spotlights how Gen X’s attempt to close the wealth gap “won’t be easy,” as 32% of survey takers have all their assets in cash and, separately, only 11% will wait until age 70 to receive maximum Social Security benefits.</b> Fear is the alleged primary reason for these trends, as Gen X worries they’ll lose all their cash or Social Security will soon become insolvent. |
| What are some potential spinoffs if X-Men ’97 is successful, and in what ways may these new shows be significant to the new generation of viewers? | None of the other shows have been demanded by fans to have revivals, and this speaks to these heroes’ popularity in the comics, as well. <b>Much like on the printed page, the success of X-Men ’97 can result in a spinoff cartoon for one of the new teams introduced in the series. The best bet might be a New Mutants or Generation X animated series that focuses on the younger heroes.</b> Likewise, it can be to a new generation of viewers what X-Men: The Animated Series was to kids in the 1990s.                                                                                                                                                    |
| What actions have led to an individual, a group, or a government being accused of tipping towards fascism?                                         | They want to modify history. <b>They want to ban books. And they wanna do all this stuff that’s like, yeah, you’re tipping towards fascism,” he said.</b> Compared with boomers or millennials, Generation X is rarely invoked in national political debates.                                                                                                                                                                                                                                                                                                                                                                                                      |

## Building contexts

### Sentence Window Retrieval

## Evaluation
