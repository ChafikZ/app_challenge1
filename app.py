import os

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go

# K-means function
from sklearn.cluster import KMeans

# Function to standardize the data 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Functions for hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

df_foot = pd.read_csv('footdata.csv')
df = pd.read_csv('team_win_bet.csv')
data_villes = pd.read_csv('data_villes.csv')

home_shots = df_foot[['HomeTeam', 'FTHG', 'HS', 'HST']] #buts, tirs, tirs cadrés E1
away_shots = df_foot[['AwayTeam', 'HTAG', 'AS', 'AST']] #buts, tirs, tirs cadrés E2
home_shots.columns = ['team', 'buts', 'tirs', 'tirs cadrés']
away_shots.columns = ['team', 'buts', 'tirs', 'tirs cadrés']
total_shots = pd.concat([home_shots, away_shots]) #on combine les deux dataframes pour avoir les stats totals domicile et ext
total_shots = total_shots.groupby('team').sum() #nb de buts, tirs et tirs cadrés pour chaque équipe

total_shots = total_shots.reset_index()

teams = df_foot.HomeTeam.unique()




df_HA_win = df_foot.filter(['HomeTeam','AwayTeam','B365H','B365D','B365A','FTR'])
df_HA_win["BetTeam"] = ''
df_HA_win['BetTeam'] = df_HA_win.iloc[0:][['B365H', 'B365D', 'B365A']].idxmin(axis=1)
df_HA_win['BetTeam'] = df_HA_win['BetTeam'].str.replace(r'B365', '')


df_team_win_bet = pd.DataFrame(columns = ['Team','nb_win','nb_good_bet'])

for i in range(0,len(teams)):
    nbwinH_team = len(df_HA_win[(df_HA_win['HomeTeam']==teams[i]) & (df_HA_win['BetTeam'] =='H')])
    nbwinA_team = len(df_HA_win[(df_HA_win['AwayTeam']==teams[i]) & (df_HA_win['BetTeam'] =='A')])
    nbwin_team = nbwinH_team + nbwinA_team

    nbwinBH_team = len(df_HA_win[(df_HA_win['HomeTeam']==teams[i]) & (df_HA_win['BetTeam'] =='H') & (df_HA_win['BetTeam']=='H')])
    nbwinBA_team = len(df_HA_win[(df_HA_win['AwayTeam']==teams[i]) & (df_HA_win['BetTeam'] =='H') & (df_HA_win['BetTeam']=='A')])
    nbwinB_team = nbwinBH_team + nbwinBA_team
    df_team_win_bet = df_team_win_bet.append({'Team': teams[i], 'nb_win': nbwin_team, 'nb_good_bet': nbwinB_team}, ignore_index=True)

data_villes = data_villes.set_index(data_villes.iloc[:,0])
data_villes = data_villes.drop(data_villes.columns[0], axis=1)
X2 = data_villes.values

std_scale2 = StandardScaler().fit(X2)
X_scaled2 = std_scale2.transform(X2)
pca = PCA().fit(X_scaled2[:,:12])

X_proj2 = pca.transform(X_scaled2[:,:12])
	
kmeans_multiple = KMeans(n_clusters=2,n_init=40,init='random').fit(X_proj2)
centers=kmeans_multiple.cluster_centers_

test = []
for i in range(len(kmeans_multiple.labels_)):
    if kmeans_multiple.labels_[i] == 1:
        test.append('red')
    else:
        test.append('green')
		
# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
    'background-color': '#f8f9fa'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '25%',
    'margin-right': '5%',
    'padding': '20px 10p'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}

controls = dbc.FormGroup(
    [
        dcc.Markdown('''
			#### Dash and Markdown

			Dash supports [Markdown](http://commonmark.org/help).

			Markdown is a simple way to write and format text.
			It includes a syntax for things like **bold text** and *italics*,
			[links](http://commonmark.org/help), inline `code` snippets, lists,
			quotes, and more.
			'''),
    ]
)

sidebar = html.Div(
    [
        html.H2('Informations sur la démarche', style=TEXT_STYLE),
        html.Hr(),
        controls
    ],
    style=SIDEBAR_STYLE,
)

content_first_row = dbc.Row([
    dbc.Col(
        md=2
    ),
    dbc.Col(
        dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H4("Membres de l'équipe", className='card-title', style=CARD_TEXT_STYLE),
                        html.P('Guillaume Chiquet, Fabien Dufay, Soriba Diabi, Chafik Zerrouki', style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=3
    ),
    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Présentation du jeu de données', className='card-title', style=CARD_TEXT_STYLE),
                        html.P('Notre jeu de données contient l’ensemble des matchs du championnat de France de Football 2007-2008. Ce championnat est composé de 20 équipes qui s’affrontent lors des matchs allers-retours.', style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=4
    ),
    dbc.Col(
        md=3
    )
])

content_second_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(figure =  px.imshow(data_villes.corr(), color_continuous_scale=px.colors.sequential.Viridis) 
							), md=4
        ),
        dbc.Col(
            dcc.Graph(figure = px.bar(df, x="Team", y="nb_win").update_xaxes(categoryorder='total descending')), md=4
        ),
        dbc.Col(
             md=4
        )
    ]
)

content_text = dbc.Row([
    dbc.Col(
        md=2
    ),

    dbc.Col(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4('Analyse jeu de donnée', className='card-title', style=CARD_TEXT_STYLE),
                        html.P("Au cours de ce championnat il y a eu 167 victoires à domicile, 97 à l'extérieur et 116 matchs nuls.", style=CARD_TEXT_STYLE),
                    ]
                ),
            ]

        ),
        md=8
    ),
    dbc.Col(
        md=2
    )
])

content_third_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(figure = go.Figure(data=[
				go.Bar(name='Buts', x=total_shots['team'], y=total_shots['buts']),
				go.Bar(name='Tirs', x=total_shots['team'], y=total_shots['tirs']),
				go.Bar(name='Tirs cadrés', x=total_shots['team'], y=total_shots['tirs cadrés']),
			])
			
			)
			
			
			, md=12,
        )
    ]
)

content_fourth_row = dbc.Row(
    [
        dbc.Col(
            dcc.Graph(id='graph_5'), md=6
        ),
        dbc.Col(
            dcc.Graph(id='graph_6'), md=6
        )
    ]
)

content = html.Div(
    [
        html.H1('Challenge 1 - IMT Atlantique', style=TEXT_STYLE),
        html.Hr(),
        content_first_row,
        content_second_row,
		content_text,
        content_third_row,
        content_fourth_row
    ],
    style=CONTENT_STYLE
)

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([sidebar, content])

server = app.server







if __name__ == '__main__':
    app.run_server(debug=True)