import re
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import networkx as nx

class PlayAnalysis:
    def __init__(self, play_text):
        self.play_text = play_text
        self.dialogue = self.extract_dialogue_and_narrative()
        self.df = self.create_dataframe()

    def extract_dialogue_and_narrative(self):
        sections = re.split(r'(ACT [IVX]+|EPILOGUE)', self.play_text)
        sections = [s.strip() for s in sections if s.strip()]  # Remove empty strings

        dialogue = []
        current_act = "Prologue"  # Default section
        current_character = "Narrator"

        for i, section in enumerate(sections):
            if re.match(r'ACT [IVX]+|EPILOGUE', section):
                current_act = section  # Update current act/epilogue
                continue

            lines = section.splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                match = re.match(r'^([A-Z\s]+):', line)
                if match:
                    current_character = match.group(1).strip()
                    speech = line[len(current_character) + 1:].strip()
                    dialogue.append((current_act, current_character, speech))
                else:
                    dialogue.append((current_act, "Narrator", line.strip()))

        return dialogue

    def create_dataframe(self):
        df = pd.DataFrame(self.dialogue, columns=['Act', 'Character', 'Speech'])
        df['Polarity'] = df['Speech'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['Subjectivity'] = df['Speech'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        return df

    def plot_subjectivity_evolution(self):
        plt.figure(figsize=(10, 6))
        for character in self.df['Character'].unique():
            character_df = self.df[self.df['Character'] == character]
            plt.plot(character_df.index, character_df['Subjectivity'], label=character)

        plt.title('Evolution of Subjectivity by Character')
        plt.xlabel('Line Number')
        plt.ylabel('Subjectivity')
        plt.legend(title='Character')
        plt.grid(True)
        plt.show()

    def plot_polarity_evolution(self):
        plt.figure(figsize=(10, 6))
        for character in self.df['Character'].unique():
            character_df = self.df[self.df['Character'] == character]
            plt.plot(character_df.index, character_df['Polarity'], label=character)

        plt.title('Evolution of Polarity by Character')
        plt.xlabel('Line Number')
        plt.ylabel('Polarity')
        plt.legend(title='Character')
        plt.grid(True)
        plt.show()

class InteractionNetwork:
    def __init__(self, df):
        self.df = df
        self.graph = self.create_graph()

    def create_graph(self):
        G = nx.Graph()
        for character in self.df["Character"].unique():
            G.add_node(character)

        for i in range(1, len(self.df)):
            character1 = self.df.at[i - 1, "Character"]
            character2 = self.df.at[i, "Character"]

            if character1 != character2:
                weight = self.df.at[i, "Polarity"]
                act = self.df.at[i, "Act"]

                if G.has_edge(character1, character2):
                    G[character1][character2]["weight"] += weight
                    G[character1][character2]["acts"].add(act)
                else:
                    G.add_edge(character1, character2, weight=weight, acts={act})

        return G

    def visualize_network(self):
        edge_labels = {
            (u, v): f"{d['weight']} ({', '.join(sorted(d['acts']))})"
            for u, v, d in self.graph.edges(data=True)
        }

        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=3000,
            node_color="lightblue",
            font_size=12,
            font_weight="bold",
            edge_color="gray"
        )
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=10)

        plt.title("Character Interaction Network with Acts", size=15)
        plt.show()
