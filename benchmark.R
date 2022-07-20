library(tidyverse)

df <- read_csv("full_benchmark.csv") %>% 
  filter(!startsWith(model, "countVectorizer"),
         !startsWith(model, "fasttext")) %>%  
  bind_rows(read_csv("full_benchmark_2.csv")) %>% # %>% 
             # mutate(model = if_else(startsWith(model, "countVectorizer"), paste0("tfidf.", model), paste0("lower.", model)))) %>% 
  separate(model, into=c("model_name", "model_params"), sep="_", extra="merge", remove=F) %>% 
  mutate(model_params = replace_na(model_params, "")) 


df_normalized <- df %>% 
  group_by(dataset) %>% 
  mutate(precision = (precision - min(precision)) / (max(precision) - min(precision))) %>% 
  mutate(recall = (recall - min(recall)) / (max(recall) - min(recall))) %>% 
  ungroup()

df_normalized %>% 
  group_by(dataset) %>% 
  mutate(recall_discret = cut_interval(recall, 10)) %>% 
  group_by(model, recall_discret) %>% 
  summarise(mean_precision = mean(precision), recall=mean(recall), 
            model_name=model_name) %>% 
  ggplot() +
  geom_line(aes(x=recall, y=mean_precision, color=model_name, group=model),
            size=1, alpha=0.5)

df %>% 
  ggplot() +
  geom_line(aes(x=recall, y=precision, color=model_name, group=model),
            size=1, alpha=0.5) +
  facet_wrap(~dataset)

df_normalized %>% 
  ggplot() +
  geom_line(aes(x=recall, y=precision, color=model_name, group=model),
            size=1, alpha=0.5) +
  facet_wrap(~dataset)

df %>% 
  filter(model_name == "countVectorizer") %>% 
  filter(str_detect(model_params, "char_wb")) %>% 
  filter(!str_detect(model_params, "l2")) %>% 
  ggplot() +
  geom_jitter(aes(x=recall, y=precision, color=model_params), width=0.1) +
  facet_wrap(~dataset)


df %>% 
  filter((model_name == "countVectorizer" & str_detect(model_params, "char_wb") & str_detect(model_params, "cosine") & str_detect(model_params, "3")) | (model_name == "autofj")) %>% 
  ggplot() +
  geom_line(aes(x=recall, y=precision, color=model), size=2) +
  facet_wrap(~dataset)


df %>% 
  ggplot() +
  geom_line(aes(x=recall, y=precision, color=model)) +
  facet_wrap(~dataset)



df %>% 
  ggplot() +
  geom_line(aes(x=recall, y=precision, color=model_name, linetype=model_params)) +
  facet_wrap(~dataset)


df %>% 
  ggplot() +
  geom_point(aes(x=recall, y=precision, color=f1)) +
  facet_wrap(~dataset)


df %>%
  ggplot() +
  geom_jitter(aes(y = dataset, x = f1, color=model_name), width = 0, alpha=0.3)
