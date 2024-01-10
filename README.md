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

| resp                                                                                                 |
|:-----------------------------------------------------------------------|
| What policies of President Biden are causing concern among Generation X?                             |
| How has President Biden’s approval rating shifted among Generation X since his inauguration?         |
| What specific issues are driving Generation X’s skepticism towards President Biden’s administration? |
| How did Generation X vote in the 2020 Election and what might that mean for 2024?                    |
| Can President Biden regain support from Generation X before the 2024 election?                       |

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

| resp                                                                                                                                       |
|:-----------------------------------------------------------------------|
| What are the reasons for Generation X’s skepticism towards President Biden as we approach the 2024 Presidential Election?                  |
| Why does Generation X show reluctance towards President Biden’s potential re-election in 2024?                                             |
| What elements make Generation X apprehensive about the possibility of President Biden seeking re-election in 2024?                         |
| How does President Biden’s impact on issues significant to Generation X influence their sentiments towards the 2024 Presidential Election? |
| Why isn’t President Biden striking a positive chord with Generation X, especially looking forward to the 2024 Presidential Election?       |

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

| resp                                                                                                                                                                                                                                                                        |
|:-----------------------------------------------------------------------|
| Generation X may be wary of President Biden due to concerns regarding the economy and deficit spending, his handling of the pandemic, and his stance on issues such as foreign policy and climate change. Also, his age is a concern for some Gen X voters looking at 2024. |

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
set.seed(11)
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

|                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|:---------------|:-------------------------------------------------------|
| What are some characteristics of Generation X films and how did the Generation X movies “Before Sunset” and “Heathers” portray these attributes? | Travel, spontaneity, and the potential of something beautiful, especially when it comes to love, are hallmarks of Generation X films. <b>Not many did it better than the movie that spawned two more films: Before Sunset (2004) and Before Midnight (2013). “My teen angst bulls—t has a body count.”</b> If that doesn’t sum up the brilliantly dark and humorous Heathers, we don’t know what does.                                                                                                                                                                     |
| How did the inflation-adjusted cost of attending a public four-year college impact the debt situation of Generation X?                           | According to the Education Data Initiative (EDI), the inflation-adjusted cost of attending a public four-year college soared from $2,499 in 1969-70 to $9,580 in 2020-21. <b>Gen Xers, unable to afford the degrees that many emerging new-economy jobs required, became the first generation to borrow its way into school — and decades of debt. “Gen X had to pay much more for college and was more likely to have student debt,” said Rogers.</b> Despite comprising just 20% of the population, Gen X now holds nearly 57% of America’s $1.63 trillion student debt. |
| How did the criticism that Generation X faced in the ’90s compare to the criticism that Millennials and Gen Z receive from Boomers today?        | Aside from this whole “Karen generation” blip, Gen X continues to be largely overlooked, and that fact — as well as their silent delight in it — is possibly one of the most Generation X things to happen to the class of 1965 to 1980. <b>Back in the ’90s, Gen X bore the same kind of criticism Boomers tend to heap on Millennials and Gen Z now. It’s not necessarily that they want to watch a cage match.</b> It’s just they’re so relieved it’s someone else being called slackers and downers for a change.                                                      |

## Building contexts

### Sentence Window Retrieval

## Evaluation
