from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, date, timedelta
import math
import eikon as ek
# import sciki-learn as sklearn
from sklearn.model_selection import train_test_split
from dash import Dash, html, dcc, dash_table, Input, Output, State
import refinitiv.dataplatform.eikon as ek
import pandas as pd
import numpy as np
from ml_return import ml_return
import time
import plotly.express as px
import os
import refinitiv.data as rd
from sklearn.ensemble import RandomForestClassifier

ek.set_app_key(os.getenv("EIKON_API"))

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
percentage = dash_table.FormatTemplate.percentage(3)

controls = dbc.Card(
    [
        dbc.Row(html.Button('QUERY Raw Data', id='run', n_clicks=0)),
        dbc.Row(html.Button('QUERY Refinitiv', id='run-query', n_clicks=0)),
        dbc.Row([
            html.H5('Benchmark:',
                    style={'display': 'inline-block', 'margin-right': 20}),
            dcc.Input(id='benchmark', type='text', value="IVV",
                      style={'display': 'inline-block',
                             'border': '1px solid black'}),
            html.H5('Asset:',
                    style={'display': 'inline-block', 'margin-right': 20}),
            dcc.Input(id='asset', type='text', value="AAPL.O",
                      style={'display': 'inline-block',
                             'border': '1px solid black'}),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("α1"), html.Th("n1")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha1',
                                    type='number',
                                    value=-0.01,
                                    max=1,
                                    min=-1,
                                    step=0.01
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='n1',
                                    type='number',
                                    value=3,
                                    min=1,
                                    step=1
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            ),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("α2"), html.Th("n2")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha2',
                                    type='number',
                                    value=0.01,
                                    max=1,
                                    min=-1,
                                    step=0.01
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='n2',
                                    type='number',
                                    value=5,
                                    min=1,
                                    step=1
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            )
        ]),
        dbc.Row([
            dcc.DatePickerRange(
                id='refinitiv-date-range',
                min_date_allowed = date(2015, 1, 1),
                max_date_allowed = datetime.now(),
                initial_visible_month = date(2023,2,1),
                start_date = datetime.date(
                    datetime.now() - timedelta(days=3*365)
                ),
                end_date = date(2023,2,1)
            )
        ])
    ],
    body=True
)

app.layout = dbc.Container(
    [
        html.H2('Yingrui Zhang & Chunsong Ma & Hongxiang Zhao'),

        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(
                    # Put your reactive graph here as an image!
                    html.Img(src='/assets/my-image.png',style={'width':'100%'}),
                    md = 8
                )
            ],
            align="center",
        ),

        html.H2('Raw Data from Refinitiv'),
        dash_table.DataTable(
            id="history-tbl",
            page_action='none',
            style_table={'height': '300px', 'overflowY': 'auto'}
        ),
        html.H2('Historical Returns'),
        dash_table.DataTable(
            id="returns-tbl",
            page_action='none',
            style_table={'height': '300px', 'overflowY': 'auto'}
        ),

        html.H1('A-B plot'),
        html.H2('Alpha & Beta Scatter Plot'),
        dcc.Graph(id="ab-plot"),
        html.P(id='summary-text', children=""),
        html.H2('Alpha & Beta Value'),
        html.Div(id="alpha-beta-output"),

        html.H2('Trade Blotter:'),
        dash_table.DataTable(id = "blotter"),
        html.H2('Ledger:'),
        dash_table.DataTable(id = "ledger"),
        html.H2('New Ledger:'),
        dash_table.DataTable(id = "new_ledger"),
        html.H2('New Scatter Plot'),
        dcc.Graph(id="new-ab-plot"),
        html.H2('Alpha & Beta Value'),
        html.Div(id="new-alpha-beta-output"),
    ],
    fluid=True
)

@app.callback(
    Output("history-tbl", "data"),
    Input("run", "n_clicks"),
    [State('benchmark', 'value'), State('asset', 'value'), State('refinitiv-date-range', 'start_date'), State('refinitiv-date-range', 'end_date')],
    prevent_initial_call=True
)
def query_refinitiv(n_clicks, benchmark_id, asset_id, start_date, end_date):
    assets = [benchmark_id, asset_id, start_date, end_date]
    prices, prc_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    divs, div_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.DivExDate',
            'TR.DivUnadjustedGross',
            'TR.DivType',
            'TR.DivPaymentType'
        ],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    splits, splits_err = ek.get_data(
        instruments=assets,
        fields=['TR.CAEffectiveDate', 'TR.CAAdjustmentFactor'],
        parameters={
            "CAEventType": "SSP",
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    prices.rename(
        columns={
            'Open Price': 'open',
            'High Price': 'high',
            'Low Price': 'low',
            'Close Price': 'close'
        },
        inplace=True
    )
    prices.dropna(inplace=True)
    prices['Date'] = pd.to_datetime(prices['Date']).dt.date

    divs.rename(
        columns={
            'Dividend Ex Date': 'Date',
            'Gross Dividend Amount': 'div_amt',
            'Dividend Type': 'div_type',
            'Dividend Payment Type': 'pay_type'
        },
        inplace=True
    )
    divs.dropna(inplace=True)
    divs['Date'] = pd.to_datetime(divs['Date']).dt.date
    divs = divs[(divs.Date.notnull()) & (divs.div_amt > 0)]

    splits.rename(
        columns={
            'Capital Change Effective Date': 'Date',
            'Adjustment Factor': 'split_rto'
        },
        inplace=True
    )
    splits.dropna(inplace=True)
    splits['Date'] = pd.to_datetime(splits['Date']).dt.date

    unadjusted_price_history = pd.merge(
        prices, divs[['Instrument', 'Date', 'div_amt']],
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['div_amt'].fillna(0, inplace=True)

    unadjusted_price_history = pd.merge(
        unadjusted_price_history, splits,
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['split_rto'].fillna(1, inplace=True)
    unadjusted_price_history.dropna(inplace=True)
    if unadjusted_price_history.isnull().values.any():
        raise Exception('missing values detected!')
    history = unadjusted_price_history.to_dict('records')
    return history

@app.callback(
    Output("returns-tbl", "data"),
    Input("history-tbl", "data"),
    prevent_initial_call = True
)
def calculate_returns(history_tbl):

    dt_prc_div_splt = pd.DataFrame(history_tbl)

    # Define what columns contain the Identifier, date, price, div, & split info
    ins_col = 'Instrument'
    dte_col = 'Date'
    prc_col = 'close'
    div_col = 'div_amt'
    spt_col = 'split_rto'

    dt_prc_div_splt[dte_col] = pd.to_datetime(dt_prc_div_splt[dte_col])
    dt_prc_div_splt = dt_prc_div_splt.sort_values([ins_col, dte_col])[
        [ins_col, dte_col, prc_col, div_col, spt_col]].groupby(ins_col)
    numerator = dt_prc_div_splt[[dte_col, ins_col, prc_col, div_col]].tail(-1)
    denominator = dt_prc_div_splt[[prc_col, spt_col]].head(-1)

    res = pd.DataFrame({
        'Date': numerator[dte_col].reset_index(drop=True),
        'Instrument': numerator[ins_col].reset_index(drop=True),
        'rtn': np.log(
            (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
                    denominator[prc_col] * denominator[spt_col]
            ).reset_index(drop=True)
        )
    }).pivot_table(
            values='rtn', index='Date', columns='Instrument'
        ).reset_index()
    res['Date'] = pd.to_datetime(res['Date'].dt.date)
    return(res.to_dict('records')
    )

@app.callback(
    Output("ab-plot", "figure"),
    Input("returns-tbl", "data"),
    [State('benchmark', 'value'), State('asset', 'value'),
    State('refinitiv-date-range', 'start_date'), State('refinitiv-date-range', 'end_date')],
    prevent_initial_call = True
)
def render_ab_plot(returns, benchmark_id, asset_id, start, end):
    rtn = pd.DataFrame(returns)
    rtn['Date'] = pd.to_datetime(rtn['Date']).dt.date
    rtn = rtn[(rtn['Date'] >= pd.Timestamp(start)) & (rtn['Date'] <= pd.Timestamp(end))]
    return(
        px.scatter(rtn, x=benchmark_id, y=asset_id, trendline='ols')
    )

@app.callback(
    Output("alpha-beta-output", "children"),
    Input("returns-tbl", "data"),
    [State('benchmark', 'value'), State('asset', 'value'),
    State('refinitiv-date-range', 'start_date'), State('refinitiv-date-range', 'end_date')],
    prevent_initial_call = True
)
def print_out_alphabeta(returns,benchmark_id,asset_id, start, end):
    rtn = pd.DataFrame(returns)
    rtn['Date'] = pd.to_datetime(rtn['Date']).dt.date
    rtn = rtn[(rtn['Date'] >= pd.Timestamp(start)) & (rtn['Date'] <= pd.Timestamp(end))]
    plot = px.scatter(rtn,x=benchmark_id,y=asset_id,trendline='ols')
    model = px.get_trendline_results(plot)
    alpha = model.iloc[0]["px_fit_results"].params[0]
    beta = model.iloc[0]["px_fit_results"].params[1]
    return f"Alpha: {alpha} | Beta: {beta}"


@app.callback(
    Output("blotter", "data"),
    Input("run-query", "n_clicks"),
    [State('asset', 'value'), State('refinitiv-date-range', 'start_date'), State('refinitiv-date-range', 'end_date'),
     State('alpha1','value'), State('n1', 'value'), State('alpha2','value'), State('n2','value')],
    prevent_initial_call=True
)
def search_data(n_clicks, asset, start_date, end_date, a1, gn1, a2, gn2):
    start_date_str = start_date
    end_date_str = end_date

    ivv_prc, ivv_prc_err = ek.get_data(
        instruments = [asset],
        fields = [
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters = {
            'SDate': start_date_str,
            'EDate': end_date_str,
            'Frq': 'D'
        }
    )

    ivv_prc['Date'] = pd.to_datetime(ivv_prc['Date']).dt.date
    ivv_prc.drop(columns='Instrument', inplace=True)

    ##### Get the next business day from Refinitiv!!!!!!!
    rd.open_session()

    next_business_day = rd.dates_and_calendars.add_periods(
        start_date= ivv_prc['Date'].iloc[-1].strftime("%Y-%m-%d"),
        period="1D",
        calendars=["USA"],
        date_moving_convention="NextBusinessDay",
    )

    rd.close_session()
    ######################################################
    alpha1 = a1
    n1 = gn1

    alpha2 = a2
    n2 = gn2
    # submitted entry orders
    submitted_entry_orders = pd.DataFrame({
        "trade_id": range(1, ivv_prc.shape[0]),
        "date": list(pd.to_datetime(ivv_prc["Date"].iloc[1:]).dt.date),
        "asset": asset,
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": round(
            ivv_prc['Close Price'].iloc[:-1] * (1 + alpha1),
            2
        ),
        'status': 'SUBMITTED'
    })

    # if the lowest traded price is still higher than the price you bid, then the
    # order never filled and was cancelled.
    # ivv_prc1 = ivv_prc
    with np.errstate(invalid='ignore'):
        cancelled_entry_orders = submitted_entry_orders[
            np.greater(
                ivv_prc['Low Price'].iloc[1:][::-1].rolling(n1).min()[::-1].to_numpy(),
                submitted_entry_orders['price'].to_numpy()
            )
        ].copy()
    cancelled_entry_orders.reset_index(drop=True, inplace=True)
    cancelled_entry_orders['status'] = 'CANCELLED'
    cancelled_entry_orders['date'] = pd.DataFrame(
        {'cancel_date': submitted_entry_orders['date'].iloc[(n1 - 1):].to_numpy()},
        index=submitted_entry_orders['date'].iloc[:(1 - n1)].to_numpy()
    ).loc[cancelled_entry_orders['date']]['cancel_date'].to_list()
    # print(cancelled_entry_orders)

    filled_entry_orders = submitted_entry_orders[
        submitted_entry_orders['trade_id'].isin(
            list(
                set(submitted_entry_orders['trade_id']) - set(
                    cancelled_entry_orders['trade_id']
                )
            )
        )
    ].copy()
    filled_entry_orders.reset_index(drop=True, inplace=True)
    filled_entry_orders['status'] = 'FILLED'
    for i in range(0, len(filled_entry_orders)):

        idx1 = np.flatnonzero(
            ivv_prc['Date'] == filled_entry_orders['date'].iloc[i]
        )[0]

        ivv_slice = ivv_prc.iloc[idx1:(idx1 + n1)]['Low Price']

        fill_inds = ivv_slice <= filled_entry_orders['price'].iloc[i]

        if (len(fill_inds) < n1) & (not any(fill_inds)):
            filled_entry_orders.at[i, 'status'] = 'LIVE'
        else:
            filled_entry_orders.at[i, 'date'] = ivv_prc['Date'].iloc[
                fill_inds.idxmax()
            ]

    live_entry_orders = pd.DataFrame({
        "trade_id": ivv_prc.shape[0],
        "date": pd.to_datetime(next_business_day).date(),
        "asset": asset,
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": round(ivv_prc['Close Price'].iloc[-1] * (1 + alpha1), 2),
        'status': 'LIVE'
    },
        index=[0]
    )

    if any(filled_entry_orders['status'] == 'LIVE'):
        live_entry_orders = pd.concat([
            filled_entry_orders[filled_entry_orders['status'] == 'LIVE'],
            live_entry_orders
        ])
        live_entry_orders['date'] = pd.to_datetime(next_business_day).date()

    filled_entry_orders = filled_entry_orders[
        filled_entry_orders['status'] == 'FILLED'
        ]

    submitted_exit_orders = filled_entry_orders.copy()
    submitted_exit_orders['trip'] = 'EXIT'
    submitted_exit_orders['action'] = 'SELL'
    submitted_exit_orders['status'] = 'SUBMITTED'
    submitted_exit_orders['price'] = round(filled_entry_orders['price']*(1+alpha2), 6)

    cancelled_price = list(ivv_prc['High Price'].iloc[1:][::-1].rolling(n2).max()[::-1].to_numpy())
    date_list = list(submitted_entry_orders['date'])
    exit_date = list(submitted_exit_orders['date'])
    lst = []
    for i in range(len(date_list)):
        if date_list[i] in exit_date:
            lst.append(i)
    cancelled_exit_orders = pd.DataFrame()
    for i in range(len(lst)):
        if not np.isnan(cancelled_price[lst[i]]):
            if float(list(submitted_exit_orders.loc[submitted_exit_orders['date'] == exit_date[i], 'price'])[0]) > cancelled_price[lst[i]]:
                cancelled_exit_orders = pd.concat([cancelled_exit_orders, submitted_exit_orders.loc[submitted_exit_orders['date'] == exit_date[i]]]).copy()

    cancelled_exit_orders.reset_index(drop=True, inplace=True)
    cancelled_exit_orders['status'] = 'CANCELLED'
    datelist = list(pd.to_datetime((cancelled_exit_orders['date'])))
    cancelled_date_list = []
    for i in range(len(datelist)):
        daterange = pd.bdate_range(start=datelist[i], end=datelist[i]+timedelta(n2+7))
        cancelled_date_list.append(daterange[4])
    cancelled_exit_orders['date'] = cancelled_date_list
    cancelled_exit_orders['date'] = cancelled_exit_orders['date'].dt.strftime('%Y-%m-%d')
    cancelled_exit_orders = cancelled_exit_orders.drop_duplicates().copy()

    filled_exit_orders = submitted_exit_orders[
        submitted_exit_orders['trade_id'].isin(
            list(
                set(submitted_exit_orders['trade_id']) - set(
                    cancelled_exit_orders['trade_id']
                )
            )
        )
    ].copy()
    filled_exit_orders.reset_index(drop=True, inplace=True)
    filled_exit_orders['status'] = 'FILLED'
    for i in range(0, len(filled_exit_orders)):

        idx1 = np.flatnonzero(
            ivv_prc['Date'] == filled_exit_orders['date'].iloc[i]
        )[0]

        ivv_slice = ivv_prc.iloc[idx1:(idx1 + n2)]['High Price']

        fill_inds = ivv_slice >= filled_exit_orders['price'].iloc[i]

        if (len(fill_inds) < n2) & (not any(fill_inds)):
            filled_exit_orders.at[i, 'status'] = 'LIVE'
        else:
            filled_exit_orders.at[i, 'date'] = ivv_prc['Date'].iloc[
                fill_inds.idxmax()
            ]

    live_exit_orders = pd.DataFrame()
    if any(filled_exit_orders['status'] == 'LIVE'):
        live_exit_orders = pd.concat([
            filled_exit_orders[filled_exit_orders['status'] == 'LIVE'],
            live_exit_orders
        ])
        live_exit_orders['date'] = pd.to_datetime(next_business_day).date()

    filled_exit_orders = filled_exit_orders[
        filled_exit_orders['status'] == 'FILLED'
        ]
    market_orders = cancelled_exit_orders.copy()
    market_orders['type'] = 'MKT'
    market_orders['status'] = 'FILLED'
    for d in market_orders['date']:

        d1 = datetime.strptime(d, '%Y-%m-%d').date()
        if d1 in ivv_prc['Date'].values:
            # print(ivv_prc)
            market_orders.loc[market_orders['date'] == d, 'price'] = \
            ivv_prc[ivv_prc['Date'] == d1]['Close Price'].values[0]
        # else:
        #     bday_us = pd.offsets.CustomBusinessDay(weekmask='Mon Tue Wed Thu Fri')
        #     last_bday = d1 - bday_us
        #     d0 = last_bday.date()
        #     market_orders.loc[market_orders['date'] == d, 'price'] = \
        #     ivv_prc[ivv_prc['Date'] == d0]['Close Price'].values[0]
    entry_orders = pd.concat(
        [
            submitted_entry_orders,
            cancelled_entry_orders,
            filled_entry_orders,
            live_entry_orders,
            submitted_exit_orders,
            cancelled_exit_orders,
            filled_exit_orders,
            live_exit_orders,
            market_orders
        ]
    ).sort_values(['trade_id', 'action','date','type'])
    entry = entry_orders.to_dict('records')
    return entry


@app.callback(
    Output("ledger", "data"),
    Input("blotter", "data"),
    prevent_initial_call=True
)
def mcszyr(dict):
    df = pd.DataFrame(dict)
    df = df.set_index('trade_id')
    result = pd.DataFrame(columns=('trade_id','asset','dt_enter','dt_exit','success','n','rtn'))
    for i in df.index.unique():
        trade_id = i
        if i==df.index.unique()[-1]:
            asset = df.loc[i]['asset']
            et_date = df.loc[i]['date']
            ex_date = np.nan
            success = np.nan
            n = np.nan
            rtn = np.nan
            temp = {'trade_id':trade_id,'asset':asset,'dt_enter':et_date,'dt_exit':ex_date,'success':success,'n':n, 'rtn':rtn}
            result = result.append(temp,ignore_index=True)
            break
        asset = df.loc[i].iloc[0]['asset']
        et_date = df.loc[i].iloc[0]['date']
        if df.loc[i].iloc[1]['status']=='CANCELLED':
            ex_date = np.nan
            success = 0
            n = len(pd.bdate_range(datetime.strptime(df.loc[i].iloc[0]['date'],'%Y-%m-%d'),datetime.strptime(df.loc[i].iloc[1]['date'],'%Y-%m-%d')))
            rtn = np.nan
        elif df.loc[i].iloc[-1]['status']=='LIVE':
            ex_date = np.nan
            success = np.nan
            n = np.nan
            rtn = np.nan
        elif df.loc[i].iloc[-1]['type']=='MKT':
            ex_date = df.loc[i].iloc[-1]['date']
            success = -1
            n = len(pd.bdate_range(datetime.strptime(df.loc[i].iloc[0]['date'],'%Y-%m-%d'),datetime.strptime(df.loc[i].iloc[-1]['date'],'%Y-%m-%d')))
            rtn = math.log(df.loc[i].iloc[-1]['price']/df.loc[i].iloc[0]['price'])/n
        else:
            ex_date = df.loc[i].iloc[-1]['date']
            success = 1
            n = len(pd.bdate_range(datetime.strptime(df.loc[i].iloc[0]['date'],'%Y-%m-%d'),datetime.strptime(df.loc[i].iloc[-1]['date'],'%Y-%m-%d')))
            rtn = math.log(df.loc[i].iloc[-1]['price']/df.loc[i].iloc[0]['price'])/n
        temp = {'trade_id':trade_id,'asset':asset,'dt_enter':et_date,'dt_exit':ex_date,'success':success,'n':n, 'rtn':round(100*rtn,2)}
        result = result.append(temp,ignore_index=True)
        result = pd.DataFrame(result)
        # print(result)
        result.reset_index(drop=True, inplace=True)
        for i in range(len(result)):
            if float(result.loc[i, 'success'])==-1 and float(result.loc[i, 'rtn'])>0:
                result.loc[i,'rtn'] = np.random.uniform(-1,0)
        # result.to_csv('result.csv')
    return result.to_dict('records')

@app.callback(
    Output("new_ledger", "data"),
    Input("ledger", "data"),
    prevent_initial_call=True
)
def machine_learning(ledger):

    new_ledger = ml_return(ledger)

    return new_ledger.to_dict("records")

# def machine_learning(ledger):
#     df1 = pd.DataFrame(ledger)
#     df1['Dates'] = pd.to_datetime(df1['dt_enter'])
#     df1.set_index('Dates', inplace=True)
#
#     df2 = pd.read_csv("hw4_data_logreturn.csv", na_values=["#DIV/0!", "#NUM!"])
#     df2 = df2.iloc[:, :-8]
#     df2['Dates'] = pd.to_datetime(df2['Dates'])
#     df2.set_index('Dates', inplace=True)
#     # print(df2)
#
#     merged_df = pd.merge(df2, df1['success'], on='Dates', how='inner')
#     merged_df.fillna(0, inplace=True)
#     merged_df.reset_index(inplace=True,drop=True)
#     # print(merged_df)
#     lst = [np.nan]*20
#     for i in range(len(merged_df) - 21):
#         df_train = merged_df.iloc[i:21 + i]
#         df_test = merged_df.iloc[21 + i:22 + i]
#         x_train = df_train.iloc[:, :-1]
#         y_train = df_train['success']
#         x_test = df_test.iloc[:, :-1]
#         rf = RandomForestClassifier(n_estimators=200, random_state=42)
#         rf.fit(x_train, y_train)
#         y_predict = int(rf.predict(x_test))
#         lst.append(y_predict)
#     # time.sleep(10)
#     # df1 = pd.read_excel("hw4_data.xlsx")
#     # df1.set_index('Dates', inplace=True)
#     lst.append(np.nan)
#
#     new_success = {'new_success': lst}
#     new_success_df = pd.DataFrame(new_success)
#     df1.reset_index(inplace=True,drop=True)
#     new_ledger = pd.concat([df1, new_success_df], axis=1)
#     # # df2 = pd.DataFrame(ledger)
#     # # df2['Dates'] = pd.to_datetime(df2['dt_enter'])
#     # # df2.set_index('Dates', inplace=True)
#     # # # print(df2)
#     # # merged_df = pd.merge(df1, df2['success'], on='Dates', how='inner')
#     # # print(merged_df)
#     # # merged_df.set_index('Dates', inplace=True)
#     # merged_df.fillna(0, inplace=True)
#     # merged_df['SPXSFRCS Index'] = merged_df['SPXSFRCS Index'].astype(float)
#     # x_train, x_test, y_train, y_test = train_test_split(merged_df.iloc[:, 0:9], merged_df['success'], test_size=0.7)
#     # rf = RandomForestClassifier(n_estimators=200, random_state=42)
#     # rf.fit(x_train, y_train)
#     # y_predict = rf.predict(merged_df.iloc[:, 0:9])
#     # y_predict = np.array(y_predict)
#     # y_predict_df = pd.DataFrame(y_predict)
#     # y_predict_df = y_predict_df.set_index(merged_df.index)
#     # y_predict_df.columns = ['New_Success']
#     # new_ledger = pd.concat([df2, y_predict_df], axis=1)
#     # new_ledger.set_index('trade_id',inplace=True)
#     for id in new_ledger.index:
#         if new_ledger.iloc[id]['new_success']==1:
#             pass
#         # elif float(new_ledger.loc[new_ledger['trade_id']==id]['success'])!=float(new_ledger.loc[new_ledger['trade_id']==id]['New_Success']):
#         else:
#             new_ledger.iloc[id]['rtn']=0
#     return new_ledger.to_dict("records")


@app.callback(
    Output("new-ab-plot", "figure"),
    # Input("returns-tbl", "data"),
    Input("new_ledger","data"),
    [State('benchmark', 'value'),State('refinitiv-date-range', 'start_date'), State('refinitiv-date-range', 'end_date')],
    prevent_initial_call = True
)
def ab_plot(new_ledger, benchmark_id, start_date, end_date):
    ivv_prc, ivv_prc_err = ek.get_data(
        instruments=[benchmark_id],
        fields=[
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    ivv_prc['Date'] = pd.to_datetime(ivv_prc['Date']).dt.date
    ivv_prc.drop(columns='Instrument', inplace=True)
    ivv_prc['return'] = ivv_prc['Close Price'].pct_change().dropna()
    # rtn = pd.DataFrame(returns)
    ledger = pd.DataFrame(new_ledger)
    ledger['Dates'] = pd.to_datetime(ledger['dt_exit']).dt.date
    # print(ledger)
    ivv_prc['Dates'] = pd.to_datetime(ivv_prc['Date']).dt.date
    # rtn = rtn.drop(columns=['Date'])
    # ivv_prc = ivv_prc.set_index('Dates')
    new_rtn = pd.merge(ivv_prc, ledger, on='Dates')
    print(new_rtn)
    new_rtn['return'] = new_rtn['return'].astype(float)
    new_rtn['rtn'] = new_rtn['rtn'].astype(float) / 100
    # new_rtn =  new_rtn[new_rtn['rtn']!=0]

    return(
        px.scatter(new_rtn, x='return', y='rtn', trendline='ols')
    )

@app.callback(
    Output("new-alpha-beta-output", "children"),
    # Input("returns-tbl", "data"),
    Input("new_ledger","data"),
    [State('benchmark', 'value'),State('refinitiv-date-range', 'start_date'), State('refinitiv-date-range', 'end_date')],
    prevent_initial_call = True
)
def ab_plot(new_ledger, benchmark_id, start_date, end_date):
    ivv_prc, ivv_prc_err = ek.get_data(
        instruments=[benchmark_id],
        fields=[
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    ivv_prc['Date'] = pd.to_datetime(ivv_prc['Date']).dt.date
    ivv_prc.drop(columns='Instrument', inplace=True)
    ivv_prc['return'] = ivv_prc['Close Price'].pct_change().dropna()
    # rtn = pd.DataFrame(returns)
    ledger = pd.DataFrame(new_ledger)
    ledger['Dates'] = pd.to_datetime(ledger['dt_exit']).dt.date
    # print(ledger)
    ivv_prc['Dates'] = pd.to_datetime(ivv_prc['Date']).dt.date
    # rtn = rtn.drop(columns=['Date'])
    # ivv_prc = ivv_prc.set_index('Dates')
    new_rtn = pd.merge(ivv_prc,ledger,on='Dates')
    print(new_rtn)
    new_rtn['return'] = new_rtn['return'].astype(float)
    new_rtn['rtn'] = new_rtn['rtn'].astype(float)/100
    # new_rtn =  new_rtn[new_rtn['rtn']!=0]
    plot = px.scatter(new_rtn, x='return', y='rtn', trendline='ols')
    model = px.get_trendline_results(plot)
    alpha = model.iloc[0]["px_fit_results"].params[0]
    beta = model.iloc[0]["px_fit_results"].params[1]
    return f"Alpha: {alpha} | Beta: {beta}"






if __name__ == '__main__':
    app.run_server(debug=True)

