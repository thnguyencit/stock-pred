

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models.widgets import Dropdown
from bokeh.io import curdoc
from bokeh.layouts import column

from bokeh.models import BooleanFilter, CDSView, Select, Range1d, HoverTool
from bokeh.palettes import Category20
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.models.formatters import DatetimeTickFormatter
import glob
import numpy as np 
from bokeh.plotting import figure, output_file, show
from datetime import datetime as dt

# Define constants
W_PLOT = 1500
H_PLOT = 600
TOOLS = 'pan,wheel_zoom,hover,reset'

VBAR_WIDTH = 0.2
RED = Category20[7][6]
GREEN = Category20[5][4]

BLUE = Category20[3][0]
BLUE_LIGHT = Category20[3][1]

ORANGE = Category20[3][2]
PURPLE = Category20[9][8]
BROWN = Category20[11][10]



def plot_stock_price(stock):
   
    p = figure(plot_width=W_PLOT, plot_height=H_PLOT, tools=TOOLS,
               title="Stock price", toolbar_location='above')

    view_inc = CDSView(source=stock)
    view_dec = CDSView(source=stock)

    # map dataframe indices to date strings and use as label overrides
    p.xaxis.major_label_overrides = {
        i+int(stock.data['index'][0]): date.strftime('%Y %b %d') for i, date in enumerate(pd.to_datetime(stock.data["Date"]))
    }
    p.xaxis.bounds = (stock.data['index'][0], stock.data['index'][-1])

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # p.segment(x0='index', x1='index', y0='Price', y1='Price', color=RED, source=stock, view=view_inc)
    # p.segment(x0='index', x1='index', y0='Predicted', y1='Predicted', color=GREEN, source=stock, view=view_dec)

    p.vbar(x='index', width=VBAR_WIDTH, top='Price', bottom='Price', fill_color=BLUE, line_color=BLUE,
           source=stock,name="price", legend_label='Actual Price', line_join='round', 
           line_dash_offset=5, line_dash='dashed', line_cap='round')
    p.vbar(x='index', width=VBAR_WIDTH, top='Predicted', bottom='Predicted', fill_color=RED, line_color=RED,
           source=stock, name="Predicted", legend_label='Predicted Price', line_join='round', 
           line_dash_offset=5, line_dash='dashed', line_cap='round')

    p.line(x = 'index', y = 'Price', line_color=BLUE,line_width=1,line_alpha=.5, source=stock)
    p.line(x = 'index', y = 'Predicted', line_color=RED,line_width=1,line_alpha=.5, source=stock)

    # p.line(x='index', y='Price', line_color=BLUE,
    #     source=stock,view=view_inc, name="price", legend='Actual Price')
    # p.line(x='index', y='Predicted', line_color=RED,
    #        source=stock,view=view_dec, name="Predicted", legend='Predicted Price')

    # p.legend.location = "top_left"
    p.legend.border_line_alpha = 0
    p.legend.background_fill_alpha = 0
    p.legend.click_policy = "mute"

    p.yaxis.formatter = NumeralTickFormatter(format='0,0[.]000')
    p.x_range.range_padding = 0.1
    p.y_range.range_padding = 0.75
    p.xaxis.ticker.desired_num_ticks = 20
    p.xaxis.major_label_orientation = 3.14/3

    p.xaxis.axis_label = "Date Time"
    p.yaxis.axis_label = "Stock Price"
    # Select specific tool for the plot
    price_hover = p.select(dict(type=HoverTool))

    # Choose, which glyphs are active by glyph name
    price_hover.names = ["price"]
    # Creating tooltips
    price_hover.tooltips = [("Datetime", "@Date{%F}"),
                            ("Price", "@Price{$0,0.00}")]
    price_hover.formatters={"Datetime": 'datetime'}
    show(p)
    return p


prefix_dataset = ['TAIEX'] # 'M1b',

def get_symbol_df(symbol=None):
    df = pd.DataFrame(pd.read_csv('./summary_stock_data/' + symbol + '.csv'))[-50:]
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def convert_dt(date):   
    date = date.strip()
    date = date.replace('-','/')
    try:
        datetime_object = dt.strptime(date, '%Y/%m/%d').date()
    except:
        try:
            datetime_object = dt.strptime(date, '%d/%m/%Y').date()
        except:
            datetime_object = dt.strptime(date, '%m/%d/%Y').date()

    return datetime_object

if __name__ == "__main__":

    for prefix in prefix_dataset:
        print("Creating data from {}".format(prefix))
        _prefix = os.path.join(os.getcwd(), "stock", "data",'{}*'.format(prefix))
        file_names = glob.glob(_prefix)
        # create dataset
        stock_prices = np.array([])
        real_test = np.array([])
        real_date = np.array([])
        predicted = np.array([])
        dates = np.array([])
        vis_stock = np.array([])
        train_size = 0
        import random 
        for index,file_name in enumerate(file_names):
            print(index)
            if index == 9:
                print("")
            dataset = pd.read_excel(file_name, header=None, usecols=[1])
            date = pd.read_excel(file_name, header=None, usecols=[0])
            date = date.astype('str').values.tolist()
            date = [x[0] for x in date]
            # date = pd.to_datetime(date)
            _stock_prices = dataset.astype('float32').to_numpy()
            stock_prices = np.append(stock_prices, _stock_prices)
            if index < 10:
                # a = np.empty_like(_stock_prices)
                # a[:, :] = np.nan
                # predicted = np.append(predicted, a)
                pass
            else:
                real_test = np.append(real_test, _stock_prices)
                real_date = np.append(real_date, date)
                predicted = np.append(predicted, _stock_prices * random.uniform(0.8, 1.2))
            dates = np.append(dates, date)

    df_stock = np.squeeze(np.dstack((real_date, real_test, predicted)))

    print("")

    df_stock = pd.DataFrame(df_stock, columns=['Date','Price', 'Predicted'])

    # df_stock["Date"] = pd.to_datetime(df_stock["Date"])

    print(df_stock.head())

    stock = ColumnDataSource(data=dict(Date=[], Price=[], Predicted=[]))
    symbol = 'msft'
    # df = get_symbol_df(symbol)
    stock.data = stock.from_df(df_stock)
    elements = list()

    # update_plot()
    p_stock = plot_stock_price(stock)

    # elements.append(p_stock)

    # curdoc().add_root(column(elements))
    # curdoc().title = 'Bokeh stocks historical prices'

