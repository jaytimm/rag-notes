    ## [1] "2024-01-10"

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

| resp                                                                                              |
|:-----------------------------------------------------------------------|
| What specific policies of President Biden has Generation X expressed concern about?               |
| How has President Biden’s administration addressed the concerns of Generation X?                  |
| What is the importance of the Generation X vote in the 2024 Presidential Election?                |
| How have Generation X’s political views evolved during President Biden’s term?                    |
| What alternatives does Generation X seem to favor as the 2024 Presidential Election draws closer? |

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

| resp                                                                                                                   |
|:-----------------------------------------------------------------------|
| Why is Generation X skeptical about President Biden’s potential re-election in 2024?                                   |
| What are the reasons for the apprehension of Generation X towards President Biden as we approach 2024?                 |
| Why does President Biden seem to struggle with gaining the support of Generation X for the 2024 Presidential Election? |
| What are the factors contributing to the uncertainty of Generation X about President Biden’s leadership in 2024?       |
| Why is Generation X hesitant about backing President Biden in the upcoming 2024 election?                              |

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

| resp                                                                                                                                                                                                                                                                                                                           |
|:-----------------------------------------------------------------------|
| Generation X is wary due to perceived weak economic policies, high inflation, and potential erosion of personal liberties under Biden’s administration. These concerns, combined with perceived lack of effective communication and transparency, potentially make Biden a less favored candidate for Gen X in 2024 elections. |

### Hypothetical questions

1.  Use LLM to generate specific questions for each text chunk
2.  Embed chunk questions
3.  Retrieve documents via question vectors
4.  Aggregate highest ranked retrieved documents

#### Build demo corpus & text chunks

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

#### Prompt & output

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

|                                                                                                     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|:------------|:----------------------------------------------------------|
| What are some ways in which Gen X has contributed to the popularity and acceptance of hip-hop?      | “It brought people sanity during a crazy time.” <b>D-Nice finds hip-hop and helping communities to be two defining traits of Generation X. He says the “MTV generation” has grown up viewing a wide array of music, including hip-hop.</b> He says that before the 1988 debut of “Yo!                                                                                                                                                                                                          |
| What might be some benefits of Generation X starting retirement planning early according to D-Nice? | By putting a plan in place today, Generation X can work with their family to make sure the money their family has worked so hard to earn stays in their family and makes a rich retirement even more possible.” <b>Why retiring later than you hoped can be a good idea. Investors could be caught off-guard by a recession and a steep decline headed for stocks, Evercore ISI’s Julian Emanuel warned.</b> The meme coin has created a lot of crypto millionaires over the past three years. |
| How might a later retirement date be beneficial for Generation X?                                   | On the other hand, Gen X Democrats like Piccolo point their disgust at conservative efforts to curb speech, particularly in schools and libraries. <b>“You can’t change history. They want to modify history.</b> They want to ban books.                                                                                                                                                                                                                                                      |

## Building contexts

### Sentence Window Retrieval

## Evaluation
