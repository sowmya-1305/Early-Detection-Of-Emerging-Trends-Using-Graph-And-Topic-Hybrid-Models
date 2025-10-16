import argparse
import random
import csv
from datetime import datetime, timedelta

SAMPLE_TOPICS = [
    ("AI in Education", ["#AIinEducation", "ai education", "edtech"]),
    ("New Smartphone", ["#SmartphoneLaunch", "new phone", "tech"]),
    ("Football Match", ["#Football", "game", "goal"]),
    ("Climate Action", ["#ClimateAction", "climate", "sustainability"]),
    ("Movie Release", ["#NewMovie", "movie", "cinema"]),
]

def random_text(topic_words):
    templates = [
        "Breaking: {} just announced!",
        "Discussion on {} — what do you think?",
        "Many reports about {}. Trending now.",
        "I love the updates around {}",
        "{} seems to be changing the industry."
    ]
    t = random.choice(templates)
    return t.format(random.choice(topic_words))

def main(outfile, n):
    start = datetime.utcnow() - timedelta(days=1)
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tweet_id","user_id","text","timestamp","hashtags","mentions","retweet_count","reply_count"])
        for i in range(n):
            topic, tags = random.choice(SAMPLE_TOPICS)
            text = random_text(tags)
            ts = start + timedelta(seconds=random.randint(0, 86400))
            hashtags = random.choice(tags)
            mentions = f"user{random.randint(1,150)}"
            writer.writerow([f"t{i}", f"user{random.randint(1,1000)}", text, ts.isoformat(), hashtags, mentions, random.randint(0,50), random.randint(0,20)])
    print(f"Generated {n} fake tweets to {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", default="data/fake_tweets.csv")
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()
    main(args.outfile, args.n)

"""
Collect data stub. Replace with actual Twitter API v2 or Pushshift code.
Example function: fetch_tweets(query, start_time, end_time, bearer_token)
"""
def collect_from_twitter_stub(query, max_results=100):
    # Replace with tweepy or `requests` calls to Twitter API v2 filtered stream
    raise NotImplementedError("Replace this stub with real data collection using Twitter API v2.")
"""Preprocessing: read CSV tweets -> cleaned dataframe"""
import argparse
import pandas as pd
import re
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

def clean_text(text):
    text = re.sub(r"http\S+","", text)
    text = re.sub(r"@\w+","", text)
    text = re.sub(r"#","", text)
    text = re.sub(r"[^0-9A-Za-z\s]", "", text)
    text = text.lower().strip()
    return text

def preprocess(df):
    texts = []
    for doc in tqdm(df['text'].astype(str).tolist(), desc="Cleaning"):
        t = clean_text(doc)
        toks = [token.lemma_ for token in nlp(t) if not token.is_stop and token.is_alpha]
        texts.append(" ".join(toks))
    df['clean_text'] = texts
    return df

def main(input, out):
    df = pd.read_csv(input, parse_dates=['timestamp'])
    df = preprocess(df)
    df.to_csv(out, index=False)
    print(f"Wrote cleaned data to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="data/cleaned.csv")
    args = parser.parse_args()
    main(args.input, args.out)

"""Run BERTopic on cleaned texts and output topic assignments and topic centroids."""
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import numpy as np

def run_bertopic(input_csv, out_csv):
    df = pd.read_csv(input_csv)
    texts = df['clean_text'].fillna("").astype(str).tolist()

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=True)

    topic_model = BERTopic(embedding_model=embedder, verbose=True)
    topics, probs = topic_model.fit_transform(texts, embeddings)

    df['topic'] = topics
    df.to_csv(out_csv, index=False)

    # Save topic info
    topics_info = topic_model.get_topic_info()
    topics_info.to_csv(out_csv.replace(".csv","_info.csv"), index=False)
    print(f"Saved topics to {out_csv} and topic info to {out_csv.replace('.csv','_info.csv')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="data/topics.csv")
    args = parser.parse_args()
    run_bertopic(args.input, args.out)

"""Build user-interaction graph from cleaned tweets. Save as pickle."""
import argparse
import pandas as pd
import networkx as nx
import pickle
from tqdm import tqdm
import ast

def build_graph(input_csv, out_pkl):
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])
    G = nx.DiGraph()
    # Add users as nodes
    users = df['user_id'].unique()
    G.add_nodes_from(users)
    # For demo we use mentions column: build edges user -> mentioned_user
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graph"):
        src = row['user_id']
        mentioned = row.get('mentions', None)
        if pd.isna(mentioned) or len(str(mentioned))==0:
            continue
        # mentions in fake data are like "user123"
        tgt = str(mentioned)
        if not G.has_node(tgt):
            G.add_node(tgt)
        if G.has_edge(src,tgt):
            G[src][tgt]['weight'] += 1
        else:
            G.add_edge(src,tgt, weight=1)
    # compute PageRank
    pr = nx.pagerank(G, weight='weight')
    nx.set_node_attributes(G, pr, 'pagerank')
    with open(out_pkl, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {out_pkl} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="data/graph.pkl")
    args = parser.parse_args()
    build_graph(args.input, args.out)

"""Compute TrendScore per topic using topic novelty + community growth + pagerank delta."""
import argparse
import pandas as pd
import pickle
import numpy as np
from collections import defaultdict

def compute_topic_novelty(df):
    # Simplified novelty: compare topic frequency across two halves of dataset
    df = df.sort_values('timestamp')
    mid = len(df)//2
    first = df.iloc[:mid]
    second = df.iloc[mid:]
    freq1 = first['topic'].value_counts(normalize=True)
    freq2 = second['topic'].value_counts(normalize=True)
    topics = set(df['topic'].unique())
    novelty = {}
    for t in topics:
        p = freq1.get(t, 1e-9)
        q = freq2.get(t, 1e-9)
        novelty[t] = np.log((q+1e-9)/(p+1e-9))  # log change proxy
    # normalize to 0-1
    vals = np.array(list(novelty.values()))
    if vals.ptp()==0:
        for k in novelty: novelty[k]=0.0
    else:
        mn, mx = vals.min(), vals.max()
        for k in novelty:
            novelty[k] = (novelty[k]-mn)/(mx-mn)
    return novelty

def compute_community_growth(df, graph):
    # Simplified grouping: for each topic, gather users, compute average pagerank
    topic_users = defaultdict(set)
    for _, row in df.iterrows():
        topic_users[row['topic']].add(row['user_id'])
    growth = {}
    avg_pr = {}
    for t, users in topic_users.items():
        prs = []
        for u in users:
            if graph.has_node(u):
                prs.append(graph.nodes[u].get('pagerank', 0.0))
        avg = np.mean(prs) if prs else 0.0
        avg_pr[t] = avg
        growth[t] = len(users)
    # Normalize growth and avg_pr
    gvals = np.array(list(growth.values())).astype(float)
    pvals = np.array(list(avg_pr.values())).astype(float)
    if gvals.ptp()==0:
        gnorm = {k:0.0 for k in growth}
    else:
        gmin,gmax=gvals.min(),gvals.max()
        gnorm = {k:(growth[k]-gmin)/(gmax-gmin) for k in growth}
    if pvals.ptp()==0:
        pnorm = {k:0.0 for k in avg_pr}
    else:
        pmin,pmax=pvals.min(),pvals.max()
        pnorm = {k:(avg_pr[k]-pmin)/(pmax-pmin) for k in avg_pr}
    return gnorm, pnorm

def compute_trends(topics_csv, graph_pkl, out_csv, alpha=0.5, beta=0.3, gamma=0.2):
    df = pd.read_csv(topics_csv, parse_dates=['timestamp'])
    with open(graph_pkl,'rb') as f:
        G = pickle.load(f)
    novelty = compute_topic_novelty(df)
    growth, pagerank_norm = compute_community_growth(df, G)
    rows = []
    for t in novelty.keys():
        score = alpha*novelty.get(t,0.0) + beta*growth.get(t,0.0) + gamma*pagerank_norm.get(t,0.0)
        rows.append({"topic": t, "novelty": novelty.get(t,0.0), "growth": growth.get(t,0.0), "pagerank": pagerank_norm.get(t,0.0), "trend_score": score})
    out = pd.DataFrame(rows).sort_values("trend_score", ascending=False)
    out.to_csv(out_csv, index=False)
    print(f"Saved trend scores to {out_csv}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", default="data/trends.csv")
    args = parser.parse_args()
    compute_trends(args.topics, args.graph, args.out)

"""Basic evaluation: print top-K trends and basic stats."""
import argparse
import pandas as pd

def main(topics, trends):
    tdf = pd.read_csv(trends)
    print("Top detected trends:")
    print(tdf.head(10).to_string(index=False))
    # simple metrics: how many topics and score distribution
    print("\nSummary stats:")
    print(tdf['trend_score'].describe())

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", required=False)
    parser.add_argument("--trends", required=True)
    args = parser.parse_args()
    main(args.topics, args.trends)
