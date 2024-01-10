# RAG systems in R

Notes and thoughts.

-   [RAG systems in R](#rag-systems-in-r)
    -   [Query expansion](#query-expansion)
    -   [Building contexts](#building-contexts)
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

| resp                                                                                          |
|:-----------------------------------------------------------------------|
| What economic policies of President Biden concern Generation X?                               |
| How does President Biden’s stance on healthcare affect Generation X’s view of his presidency? |
| What role does Generation X play in the 2024 presidential election?                           |
| How does President Biden’s foreign policy relate to Generation X’s political interests?       |
| In what ways are Generation X’s expectations for the presidency not being met by Biden?       |

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

| resp                                                                                                                 |
|:-----------------------------------------------------------------------|
| What is causing Generation X’s apprehension towards President Biden as we get closer to the 2024 election?           |
| Why does Generation X seem to lack confidence in President Biden ahead of the 2024 Presidential Elections?           |
| Why is Generation X displaying signs of unease about President Biden’s leadership as the 2024 election approaches?   |
| Why does the prospect of a Biden re-election in 2024 stir concerns among the Generation X demographic?               |
| What is causing President Biden’s seeming loss of trust among Generation X voters as the 2024 election comes nearer? |

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
| Generation X’s skepticism towards President Biden for the 2024 election may stem from issues like concerns over his age, dissatisfaction with his handling of key issues such as the economy and COVID-19, and a perceived lack of progress on promised policy initiatives. |

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

|                                                                                                         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|:------------|:----------------------------------------------------------|
| What is Johnson’s career trajectory and what notable projects has he worked on?                         | But it had an impact in the media of its day – a metaphor for the arc of Johnson’s very Gen X career, which began in the heyday of print media and flourishes in today’s social media age. <b>Johnson has now logged nearly two decades with Geto & de Milly, a Manhattan public affairs firm. He made a splash alongside Beyoncé, promoting the opening of Brooklyn’s marquee Barclays Center.</b> “It was a big moment – bringing this world-class facility to the borough,” says Johnson, who also worked on The New York Times headquarters. |
| What are some reasons that delaying retirement could be beneficial for Generation X?                    | By putting a plan in place today, Generation X can work with their family to make sure the money their family has worked so hard to earn stays in their family and makes a rich retirement even more possible.” <b>Why retiring later than you hoped can be a good idea. As you get closer to retirement, your financial strategies have to shift a bit.</b> The oldest Gen Xers will soon be old enough to retire.                                                                                                                              |
| What are some common health issues that Gen Xers might face as they age, particularly regarding vision? | Like all generations, there are particular health challenges that Gen Xers should be aware of as they grow older. <b>Here are some of those challenges and ways to deal with them. The early 40s is when even people with great vision start experiencing eye problems.</b> One of the most common conditions is presbyopia.                                                                                                                                                                                                                     |

## Building contexts

### Sentence Window Retrieval

## Evaluation
