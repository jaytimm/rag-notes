2024-01-10

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

| resp                                                                                        |
|:-----------------------------------------------------------------------|
| What are the predominant political values of Generation X?                                  |
| How might President Biden’s policies affect Generation X economically?                      |
| Has President Biden’s administration addressed the concerns of Generation X?                |
| What role might Generation X play in the 2024 Presidential Elections?                       |
| What may be the long-term impacts of Biden’s presidency on Generation X’s retirement plans? |

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

| resp                                                                                                                         |
|:-----------------------------------------------------------------------|
| What are the reasons Generation X is skeptical of President Biden for the 2024 elections?                                    |
| Why does Generation X express doubt towards President Biden’s potential re-election in 2024?                                 |
| What is causing Generation X’s reluctance to support President Biden in the 2024 elections?                                  |
| Why is President Biden failing to secure the confidence of Generation X for the 2024 Presidential race?                      |
| What factors contribute to Generation X’s hesitancy towards President Biden’s candidacy for the 2024 presidential elections? |

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

| resp                                                                                                                                                                                                                                                                                                                |
|:-----------------------------------------------------------------------|
| Generation X’s wariness towards President Biden primarily arises from concerns over rising inflation, policy direction, and perceived lack of strong, decisive leadership. The potential backlash of prolonged pandemic measures also factors into their apprehension as the 2024 Presidential Election approaches. |

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

|                                                                                              |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|:----------|:------------------------------------------------------------|
| What was the topic of discussion held by Jasmin Singer with a panel of Gen Xers and experts? | Gen Xers, who grew up with mixtapes, MTV, and significant global events, are often seen as the last to fully embrace the analog world, yet some view their skepticism and independence critically. <b>Connections guest host Jasmin Singer has gathered a panel of Gen Xers and experts to discuss their legacy, myths, and lasting impact. If you’re a Gen Xer, join us to relive your mixtape and flannel days.</b>                                                                                                                                                      |
| What are the challenges that Gen Xers faced with regards to affording college degrees?       | According to the Education Data Initiative (EDI), the inflation-adjusted cost of attending a public four-year college soared from $2,499 in 1969-70 to $9,580 in 2020-21. <b>Gen Xers, unable to afford the degrees that many emerging new-economy jobs required, became the first generation to borrow its way into school — and decades of debt. “Gen X had to pay much more for college and was more likely to have student debt,” said Rogers.</b> Despite comprising just 20% of the population, Gen X now holds nearly 57% of America’s $1.63 trillion student debt. |
| What factors are influencing this individual’s vote for Biden over Trump?                    | But as of now, he says his Biden vote is driven by a disgust at the influence that former President Donald Trump has had over the Republican Party. <b>“I don’t hate the Republican Party. I hate the MAGA wing of it,” he said, “These people are lunatics.”</b> Gen Xers are facing a cascading series of economic concerns: aging parents, raising children, saving for retirement, rising housing costs, higher food and gas prices, all hitting most acutely in middle age.                                                                                           |

## Building contexts

### Sentence Window Retrieval

## Evaluation
