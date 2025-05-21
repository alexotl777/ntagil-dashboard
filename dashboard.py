import os
from datetime import datetime
from loguru import logger

from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from clustering import WordCloudShower


# ——— Тема графиков ———
DARK_THEME = dict(
    plot_bgcolor="#162a4b",
    paper_bgcolor="#162a4b",
    font_color="#f0f3f7",
    legend=dict(font=dict(color="#f0f3f7"))
)


# ——— Загрузка данных ———
def load_full_data():
    df = pd.read_excel("tagil_comments_with_labels.xlsx")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['year'] = df['date_time'].dt.year
    return df


def load_topic_data():
    df = pd.read_excel("data_with_topic_names.xlsx")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['year'] = df['date_time'].dt.year
    df['week'] = df['date_time'].dt.to_period('W').astype(str)
    return df


def insert_linebreaks(text: str, every=6):
    if '<br>' in text: return text
    words = text.split()
    res = '<br>'.join(
        [' '.join(words[i:i + every]) for i in range(0, len(words), every) if i + every < len(words)]
    ) + '<br>'
    res += (' '.join(words[len(words) - len(words) % every: len(words)]) + '<br>' if len(words) % every != 0 else '')
    return res


# ——— Графики ———
import pandas as pd
import plotly.express as px

from statsmodels.tsa.holtwinters import ExponentialSmoothing


# def insert_linebreaks(text, every=5):
#     return '<br>'.join([text[i:i + every] for i in range(0, len(text), every)])


def create_trend_by_topic_prc(df, forecast_periods=3):
    df = df.copy()
    # Подготовка месячных данных
    df['month_start'] = pd.to_datetime(df['week'].str[:7] + '-01', errors='coerce')
    all_months = df['month_start'].dropna().sort_values().unique()
    all_topics = df['topic_name'].dropna().unique()

    idx = pd.MultiIndex.from_product([all_months, all_topics], names=['month_start', 'topic_name'])
    dfw = (
        df.groupby(['month_start', 'topic_name'])
        .size()
        .reindex(idx, fill_value=0)
        .reset_index(name='count')
    )
    dfw['total_comments'] = dfw.groupby('month_start')['count'].transform('sum')
    dfw['percent'] = (dfw['count'] / dfw['total_comments'] * 100).fillna(0).round(1)
    dfw['topic_name'] = dfw['topic_name'].apply(lambda x: insert_linebreaks(x, 5))

    # Прогноз
    last_month = all_months.max()
    future_months = pd.date_range(start=last_month + pd.offsets.MonthBegin(1), periods=forecast_periods, freq='MS')
    forecasts = []
    seasonality = 12

    for topic in all_topics:
        topic_label = insert_linebreaks(topic, 5)
        ts = dfw.loc[dfw['topic_name'] == topic_label, ['month_start', 'percent']]
        ts = ts.set_index('month_start').asfreq('MS').percent
        series = ts.dropna()

        # Выбор модели
        if len(series) >= seasonality * 2:
            model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonality,
                                         initialization_method='estimated')
            fit = model.fit(optimized=True)
            fcast = fit.forecast(forecast_periods)
        elif len(series) >= 2:
            model = ExponentialSmoothing(series, trend='add', seasonal=None, initialization_method='estimated')
            fit = model.fit(optimized=True)
            fcast = fit.forecast(forecast_periods)
        else:
            last_val = series.iloc[-1] if len(series) > 0 else 0
            fcast = pd.Series([last_val] * forecast_periods, index=future_months)

        for dt, val in fcast.items():
            forecasts.append({'month_start': dt, 'topic_name': topic_label, 'percent': round(val, 1)})

    # Создание DataFrame прогнозов
    df_fc = pd.DataFrame(forecasts)
    df_fc['count'] = None
    df_fc['total_comments'] = None
    df_fc['data_type'] = 'прогноз'
    dfw['data_type'] = 'актуальный'
    df_final = pd.concat([dfw, df_fc], ignore_index=True)

    # Визуализация
    fig = px.bar(
        df_final,
        x='month_start',
        y='percent',
        color='topic_name',
        pattern_shape='data_type',
        pattern_shape_sequence=['', '/'],
        title="Распределение комментариев по темам по месяцам с прогнозом",
        labels={
            'month_start': 'Месяц',
            'percent': 'Доля комментариев (%)',
            'topic_name': 'Тема'
        },
        barmode='stack',
        custom_data=['topic_name', 'percent', 'data_type']
    )

    fig.update_layout(
        **DARK_THEME,
        margin=dict(t=50, b=50),
        xaxis=dict(
            title="Месяц",
            gridcolor="rgba(240,243,247,0.1)",
            tickformat='%b %Y',
        ),
        yaxis=dict(
            title="Доля комментариев (%)",
            gridcolor="rgba(240,243,247,0.1)",
            range=[0, 100]
        ),
        # legend=dict(
        #     x=0.82,
        #     y=1,
        #     traceorder='normal'
        # ),
        hovermode='closest'
    )
    # Пунктирная линия начала прогноза
    start_dt = future_months[0].to_pydatetime()
    # Вертикальная линия через add_shape вместо add_vline
    fig.add_shape(
        type='line',
        x0=start_dt,
        x1=start_dt,
        y0=0,
        y1=1,
        xref='x',
        yref='paper',
        line=dict(dash='dash', color='white')
    )
    # Аннотация для начала прогноза
    fig.add_annotation(
        x=start_dt,
        y=1,
        xref='x',
        yref='paper',
        text='Начало прогноза',
        showarrow=False,
        yanchor='bottom'
    )
    for trace in fig.data:
        if isinstance(trace.name, str) and ', ' in trace.name:
            trace.name = trace.name.replace(', ', ' - ')
    return fig

def create_trend_by_topic(df):
    df = df.copy()
    df['week_start'] = pd.to_datetime(df['week'].str[:10], errors='coerce')
    all_weeks = df['week_start'].dropna().sort_values().unique()
    all_topics = df['topic_name'].dropna().unique()
    idx = pd.MultiIndex.from_product([all_weeks, all_topics], names=['week_start', 'topic_name'])
    dfw = df.groupby(['week_start', 'topic_name']).size().reindex(idx, fill_value=0).reset_index(name='count')
    dfw['topic_name'] = dfw['topic_name'].apply(lambda x: insert_linebreaks(x, 5))

    fig = px.line(dfw, x='week_start', y='count', color='topic_name',
                  title="Популярность тем по неделям", markers=True,
                  labels={
                     'topic_name': 'Тема',
                     'count': 'Количество',
                     'week_start': 'Начало недели'
                  })
    fig.update_layout(
        **DARK_THEME,
        margin=dict(t=50, b=50),
        xaxis=dict(title="Неделя", gridcolor="rgba(240,243,247,0.1)"),
        yaxis=dict(title="Количество", gridcolor="rgba(240,243,247,0.1)")
    )
    return fig

def yearly_top_topics(df):
    df2 = df.groupby(['year', 'topic_name']).size().reset_index(name='count')
    top3 = df2.sort_values(['year', 'count'], ascending=[True, False]).groupby('year').head(3)

    top3['topic_name'] = top3['topic_name'].apply(lambda x: insert_linebreaks(x, 5))

    fig = px.bar(top3, x='year', y='count', color='topic_name',
                 title="Топ-3 темы по годам", barmode='group',
                 labels={
                     'topic_name': 'Тема',
                     'count': 'Количество',
                     'year': 'Год',
                     'label': 'Настроение'
                 })
    fig.update_layout(
        **DARK_THEME,
        xaxis=dict(gridcolor="rgba(240,243,247,0.1)"),
        yaxis=dict(gridcolor="rgba(240,243,247,0.1)")
    )
    return fig


def sentiment_distribution(df):
    # Переводим label на русский
    label_map = {
        'positive': 'Позитивный',
        'negative': 'Негативный',
        'neutral': 'Нейтральный'
    }
    df = df.copy()
    df['label'] = df['label'].map(label_map)

    # Группировка и график
    cnt = df.groupby(['year', 'label']).size().reset_index(name='count')
    fig = px.pie(cnt, values='count', names='label', facet_col='year',
                 hole=0.4, title="Распределение настроения по годам",
                 labels={
                     'label': 'Тональность',
                     'count': 'Количество',
                     'year': 'Год'
                 })
    fig.update_layout(**DARK_THEME)
    fig.update_traces(
        marker=dict(line=dict(color='#162a4b', width=2)),
        textposition='inside',
        textinfo='percent+label'
    )
    return fig


# ——— Layout ———
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], meta_tags=[{"name": "viewport", "content": "width=device-width"}])
server = app.server

app.layout = dbc.Container(fluid=True, children=[

    html.H1("Настроения Тагила: Дашборд"),

    dbc.Row(
        [
            dbc.Col(
                [
                    html.Label("Диапазон дат"), html.Br(), dcc.DatePickerRange(
                        id='date-range-picker',
                        style={
                            'backgroundColor': '#162a4b',
                            'color': '#f0f3f7',
                            'border': '1px solid #0e1e3e',
                            'borderRadius': '4px',
                            'padding': '2px'
                        }
                    )
                ],
                width=6
            ),
            dbc.Col(
                [
                    html.Div(id='last-updated-dash', style={
                        'color': '#9faecb',
                        'fontSize': '0.9em',
                        'textAlign': 'right',
                        'margin': '10px 0'
                    }),
                ],
                width=6
            ),
        ]
    ),
    html.Br(),

    dbc.Tabs([

        dbc.Tab(label="Графики", children=[
            dcc.Loading(
                id="loading-filter-topic-dropdown",
                type="circle",  # можно также "default", "dot"
                color="#08aeea",
                children=dbc.Row([dbc.Col(dcc.Dropdown(
                    id='filter-topic-dropdown',
                    multi=True,
                    placeholder="Все темы",
                    style={
                        'backgroundColor': '#162a4b',
                        'color': '#f0f3f7',
                        'marginTop': '2rem',
                        'border': '1px solid #0e1e3e',
                        'borderRadius': '4px'
                    },
                    className="mb-3"
                ), width=12)])
            ),
            dcc.Loading(
                id="loading-trend-topic-graph",
                type="circle",  # можно также "default", "dot"
                color="#08aeea",
                children=dbc.Row([
                    dbc.Col(dbc.Card(
                        dcc.Graph(id='trend-topic-graph')
                    ), width=12)
                ])
            ),
            dcc.Loading(
                id="loading-trend-topic-graph-prc",
                type="circle",  # можно также "default", "dot"
                color="#08aeea",
                children=dbc.Row([
                    dbc.Col(dbc.Card(
                        dcc.Graph(id='trend-topic-graph-prc')
                    ), width=12)
                ])
            ),
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dcc.Loading(
                            id="loading-top-topics-yearly",
                            type="circle",  # можно также "default", "dot"
                            color="#08aeea",
                            children=dcc.Graph(id='top-topics-yearly')
                        )
                    ),
                    width=6
                ),
                dbc.Col(
                    dbc.Card(
                        dcc.Loading(
                            id="loading-sentiment-by-year",
                            type="circle",  # можно также "default", "dot"
                            color="#08aeea",
                            children=dcc.Graph(id='sentiment-by-year')
                        )
                    ),
                    width=6
                ),
            ])

        ]),

        dbc.Tab(label="Облако и комментарии", children=[
            html.H2("Облако слов"),
            dcc.Dropdown(
                id='topic-dropdown',
                className='dark-dropdown',
                placeholder="Выберите тему",
                style={
                    'backgroundColor': '#162a4b',
                    'color': '#f0f3f7',
                    'border': '1px solid #0e1e3e',
                    'borderRadius': '4px'
                },
            ),
            html.Br(),
            dcc.Loading(
                id="loading-wordcloud-image",
                type="circle",  # можно также "default", "dot"
                color="#08aeea",
                children=html.Img(id='wordcloud-image', style={"maxWidth": "100%", "marginTop": "1rem"}),
            ),
            html.Br(),
            html.Button("Показать комментарии", id='show-comments', n_clicks=0),
            dcc.Loading(
                id="loading-comments-output",
                type="circle",  # можно также "default", "dot"
                color="#08aeea",
                children=html.Div(id='comments-output', style={'whiteSpace': 'pre-wrap', 'marginTop': '1rem'}),
            ),
        ])

    ]),

    # ——— Хранилища ———
    dcc.Store(id='store-full-df'),
    dcc.Store(id='store-topic-df'),

    # ——— Интервал обновления ———
    dcc.Interval(id='interval-component', interval=43_200_000, n_intervals=1)

])

@app.callback(
    Output('last-updated-dash', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_timestamp(n):
    mtime = os.path.getmtime("data_with_topic_names.xlsx")
    last_updated = datetime.fromtimestamp(mtime).strftime('%d.%m.%Y %H:%M')
    return f"Данные обновлены: {last_updated}"

# ——— Обновление store каждые 12 часов ———
@app.callback(
    Output('store-full-df', 'data'),
    Output('store-topic-df', 'data'),
    Output('date-range-picker', 'min_date_allowed'),
    Output('date-range-picker', 'max_date_allowed'),
    Output('date-range-picker', 'start_date'),
    Output('date-range-picker', 'end_date'),
    Output('filter-topic-dropdown', 'options'),
    Output('topic-dropdown', 'options'),
    Input('interval-component', 'n_intervals')
)
def refresh_data(n):
    logger.info("Refresh data...")
    full_df = load_full_data()
    topic_df = load_topic_data()
    options = [{"label": t, "value": t} for t in sorted(topic_df.topic_name.dropna().unique())]

    min_date = topic_df.date_time.min()
    max_date = topic_df.date_time.max()
    return (
        full_df.to_dict('records'),
        topic_df.to_dict('records'),
        min_date, max_date, min_date, max_date,
        options,
        options
    )

# ——— Обновление графика тренда ———
@app.callback(
    Output('trend-topic-graph', 'figure'),
    Input('store-topic-df', 'data'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
    Input('filter-topic-dropdown', 'value')
)
def update_trend(topic_data, s, e, sel):
    df = pd.DataFrame(topic_data)
    dff = df[(df.date_time >= s) & (df.date_time <= e)]
    if sel:
        dff = dff[dff.topic_name.isin(sel)]
    return create_trend_by_topic(dff)

@app.callback(
    Output('trend-topic-graph-prc', 'figure'),
    Input('store-topic-df', 'data'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date')
)
def update_trend(topic_data, s, e):
    df = pd.DataFrame(topic_data)
    dff = df[(df.date_time >= s) & (df.date_time <= e)]
    return create_trend_by_topic_prc(dff)

# ——— Обновление статичных графиков ———
@app.callback(
    Output('top-topics-yearly', 'figure'),
    Output('sentiment-by-year', 'figure'),
    Input('store-full-df', 'data'),
    Input('store-topic-df', 'data'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
)
def update_static_graphs(full_data, topic_data, start, end):
    full_df = pd.DataFrame(full_data)
    full_df = full_df[(full_df.date_time >= start) & (full_df.date_time <= end)]
    topic_df = pd.DataFrame(topic_data)
    topic_df = topic_df[(topic_df.date_time >= start) & (topic_df.date_time <= end)]
    return yearly_top_topics(topic_df), sentiment_distribution(full_df)

# ——— Wordcloud ———
@app.callback(Output('wordcloud-image', 'src'),
              Input('topic-dropdown', 'value'))
def upd_wc(t):
    return WordCloudShower(pd.read_excel("data_with_topic_names.xlsx")) \
        .get_wordcloud_base64_by_topic(t) if t else None

# ——— Комментарии по теме ———
@app.callback(Output('comments-output', 'children'),
              Input('topic-dropdown', 'value'),
              Input('show-comments', 'n_clicks'))
def upd_comm(topic, n):
    if ctx.triggered_id == 'topic-dropdown':
        return ""

    if ctx.triggered_id == 'show-comments' and topic:
        comms = WordCloudShower(
            pd.read_excel("data_with_topic_names.xlsx")
        ).get_comments_by_topic(topic)[:25]

        return [
            html.Div(
                className="comment-card",
                children=[
                    html.Div(
                        style={
                            'display': 'flex',
                            'alignItems': 'center',
                            'marginBottom': '8px'
                        },
                        children=[
                            html.Span(
                                "➤",
                                style={
                                    'color': '#08aeea',
                                    'marginRight': '10px',
                                    'fontSize': '1.2em'
                                }
                            ),
                            dcc.Markdown(
                                f"**Комментарий {i + 1}**  \n{c}",
                                style={'whiteSpace': 'pre-wrap'}
                            )
                        ]
                    )
                ]
            )
            for i, c in enumerate(comms)
        ]

    return ""

# ——— Запуск ———
if __name__ == "__main__":
    app.run(debug=True)
