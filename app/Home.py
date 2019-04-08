import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# import plotly.graph_objs as go

from App import app

## nav_bar
nav_bar = html.Nav([
    html.Div([
        html.Div([
            html.Button([
                html.Span(className="icon-bar"),
                html.Span(className="icon-bar"),
                html.Span(className="icon-bar")
            ], className="navbar-toggle", type="button", **{'data-toggle': 'collapse'}, **{'data-target': '#myNavbar'}),
            html.A('Image Colorization', href="#myPage", className="navbar-brand")
        ], className="navbar-header"),
        html.Div([
            html.Ul([
                html.Li([html.A('Home', href='/Home')]),
                html.Li([html.A('U Art', href='/Colorize')]),
                html.Li([html.A([
                    html.I(className="fa fa-github")
                ], href='https://github.com/Hongyu-Li/Colorize_Grayscale_Images/')])
            ], className="nav navbar-nav navbar-right")
        ], className="collapse navbar-collapse", id="myNavbar")
    ], className="container")
], className="navbar navbar-default navbar-fixed-top")

## carousel
carousel = html.Div([
    # Indicators
    html.Ol([
        html.Li(**{'data-target': '#myCarousel'}, **{'data-slide-to': '0'}, className="active"),
        html.Li(**{'data-target': '#myCarousel'}, **{'data-slide-to': '1'})
    ], className="carousel-indicators"),
    # Wrapper for slides
    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('cover2.png'),width="100%",height='70%'),
            html.Div([
                html.H3(['Audrey Hepburn']),
                html.P(['The Angel Failing To The Earth!'])
            ], className="carousel-caption")
        ], className="item active"),
        html.Div([
            html.Img(src=app.get_asset_url('cover1.png'), width="100%",height='300%'),
            html.Div([
                html.H3(['Tianxin Park(Chang Sha)']),
                html.P(['Arts Make Our World Better！'])
            ], className="carousel-caption")
        ], className="item")        
    ], className="carousel-inner", role="listbox"),
    html.A([
        html.Span(className="glyphicon glyphicon-chevron-left", **{'aria-hidden': 'true'}),
        html.Span('Previous', className="sr-only")
    ], className="left carousel-control", href="#myCarousel", role="button", **{'data-slide': 'prev'}),
    html.A([
        html.Span(className="glyphicon glyphicon-chevron-right", **{'aria-hidden': 'true'}),
        html.Span('Next', className="sr-only")
    ], className="right carousel-control", href="#myCarousel", role="button", **{'data-slide': 'next'})
], className="carousel slide", id="myCarousel", **{'data-ride': 'carousel'})

layout = html.Div([
    nav_bar,
    carousel
])