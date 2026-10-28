Movie Recommender: Sentiment Analysis and Contextual Responses
A Streamlit app that answers "What shall we watch?" The user types the kind of movie they want, such as "a cozy comedy for a rainy night." The app suggests five movies, ranked by how closely they match the query and by how well critics received them. For each suggestion it writes a short "why you might like it," based on Rotten Tomatoes critic reviews. Users answer Yes / No / Try again. Answers are shown on the page only and are not saved.


Team: Jenny Palti and Rachel Barazani
Data
Original dataset: Rotten Tomatoes Movies and Critic Reviews Dataset (Kaggle). Licensed CC0 (public domain).


The data was scraped from the publicly available website rottentomatoes.com as of 2020-10-31. The scraping took a few days, so some fields are not fully consistent between the two files. For example, tomatometer_count, tomatometer_top_critics_count, tomatometer_fresh_critics_count, and tomatometer_rotten_critics_count in the movies file do not always match the records in the critic reviews file. This project uses the critic reviews for the Fresh rates and sentiment, so these differences do not affect the results shown here, but they are worth knowing about.


Download the two files and place them in a data folder inside the project:


data/


  rotten_tomatoes_movies.csv


  rotten_tomatoes_critic_reviews.csv


The data is not stored in this repository because the reviews file is about 226 MB.
Setup
Requires Python 3.10 or newer.


python -m venv .venv


.venv\Scripts\activate          (Windows)


source .venv/bin/activate       (macOS / Linux)


pip install -r requirements.txt


The first run downloads the models from Hugging Face (all-MiniLM-L6-v2, flan-t5-small, distilbert-base-uncased, and the SST-2 sentiment model). These need an internet connection once, and the files are cached after that.
How to run
Run these from the project folder, in order:


1. Load and split the data


python data.py --data-dir data --write-splits


Loads and cleans the data, takes a 10,000-review sample, and writes the train, validation, and test splits to the hidden .data folder.


2. Sentiment models: pretrained vs. LoRA fine-tune


python sentiment.py --data-dir data


Trains the LoRA adapter and compares it with the pretrained model on the same test reviews. Results go to outputs/sentiment_results.json.


3. Try the recommender in the terminal


python recommender.py "a cozy comedy for a rainy night"


The first run embeds the whole catalog, which takes several minutes. The embeddings are cached in .data/, so later runs are fast.


4. Evaluate on the 50 test queries


python evaluate.py


Reads references.csv and writes outputs/evaluation_results.json and outputs/evaluation_per_query.csv.


5. Run the app


streamlit run app.py


Opens in your browser, usually at http://localhost:8501.
How it works
1. Catalog. Each movie with at least one labeled critic review is one entry. Its search text is the title, genres, critics' consensus, and synopsis. 21 movies with no labeled reviews are removed.
2. Search. The query and each movie's search text are embedded with all-MiniLM-L6-v2 (sentence-transformers). The 50 closest movies are kept.
3. Ranking. Each candidate's score combines its similarity to the query with its smoothed Fresh rate: (1 - w) × similarity + w × Fresh rate, with w = 0.3. The Fresh rate is pulled toward the overall average, so movies with only a few reviews don't rank at the top.
4. Explanations. For each of the top five movies, the three critic reviews most similar to the query are selected. flan-t5-small writes one sentence based on the movie's details and those reviews. If the output is empty or repeats words, the app shows the critics' consensus instead.
5. Sentiment. The pretrained distilBERT SST-2 model is used in the app. A LoRA fine-tune of distilBERT is trained on the Rotten Tomatoes reviews and compared on the same test set.
Results
Sentiment (1,000 held-out critic reviews)


Model
	Accuracy
	Fresh precision
	Fresh recall
	Loss
	Pretrained (SST-2)
	0.833
	0.886
	0.845
	0.745
	LoRA fine-tuned
	0.802
	0.873
	0.804
	0.419
	

The pretrained model is more accurate, so the app uses it. The LoRA model has a lower loss, which means its probabilities are less overconfident. It was trained on only 2,000 reviews for two epochs.


Recommendations (50 hand-written test queries)


Metric
	Result
	Hit@1
	2% (1/50)
	Hit@5
	4% at weight 0.0; 2% at weight 0.3
	BLEU
	0.25
	ROUGE-1 F
	0.121
	ROUGE-L F
	0.098
	

Hit@k counts a hit only when the exact movie written for a query appears in the top results. Many queries fit several good films, so these numbers understate how useful the suggestions are. BLEU and ROUGE compare generated replies with one hand-written reply each, so they measure wording overlap, not quality.
Files
File
	Purpose
	app.py
	Streamlit web app
	recommender.py
	Search, ranking, and explanations
	data.py
	Loads, cleans, samples, and splits the data
	sentiment.py
	Pretrained vs. LoRA sentiment comparison
	evaluate.py
	Evaluation on the 50 test queries
	references.csv
	50 test queries and hand-written reference replies
	requirements.txt
	Python packages
	Limitations
* The model runs on a CPU, so the first run and the explanations are slow.
* flan-t5-small is small, so its explanations are sometimes generic.
* The LoRA fine-tune used a small training set and short training.
* The 50 test queries are hand-written and evaluated against one reference reply each.
* The app runs on a local computer only. Hosting it would need a plan for the data, which is too large for GitHub.
Deliverables
* Streamlit app (app.py)
* Evaluation scripts and results (evaluate.py, outputs/)
* Sentiment comparison (sentiment.py, outputs/sentiment_results.json)
* Presentation slides (3 to 5 slides)