from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import scipy.interpolate as interpol
import dash_html_components as html
import dash_core_components as dcc
from astropy.table import Table
import scipy.ndimage as ndimage
import plotly.graph_objs as go
import plotly.express as px
import functools
import numpy as np
import dash
import flask
from pathlib import Path
import sys
import os
from tablefunctions import compare_clusters

if flask.current_app:
    # flask server already exists, which is created within dasha.
    # from dasha.web.extensions.dasha import dash_app as app
    from dasha.web.extensions.dasha import dash_app as app
    from dasha.web.extensions.cache import cache
    cache_func = cache.memoize()
    app.config.external_stylesheets.append(dbc.themes.LUX)
else:
    app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
    server = app.server
    # need to make available the parent as package
    # this is already done in dasha_app.py if run from DashA
    sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())
    cache_func = functools.lru_cache
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

#directories
srcDir = Path(__file__).parent
actDir = srcDir.joinpath("ACTClusters")
scatDir = srcDir.joinpath("SecondaryCatalogs")

# Load cluster data
#catalog contents and field descriptions are here:
#https://lambda.gsfc.nasa.gov/product/act/actpol_szcluster_cat_info.cfm
#read the fits file directly into an astropy Table
#trim out all clusters below -28 dec (that's goods-south)
actCat = Table.read(actDir.joinpath('DR5_cluster-catalog.fits'))
w = np.where(actCat['decDeg'] < -28)[0]
actCat.remove_rows(w)
nRemoved = len(w)
nClusters = len(actCat)

#Add columns for ancillary data 
SecondaryCatalogs = os.listdir(scatDir) 
for i in range(len(SecondaryCatalogs)):
    SecondaryCatalog = Table.read(scatDir.joinpath(SecondaryCatalogs[i]))
    compare_clusters(actCat,SecondaryCatalog, str(SecondaryCatalogs[i]))

#List of all secondary catalogs being checked for matches
CatalogList = ['ACCEPT', 'Clash', 'Herschel', 'NIKA', 'PLANCK', 'RASSEBCS', 'REFLEX', 'Rosgalclus', 'XCS']
    
#create masks for the necessary components
catalogs = {}
names = {}
masses = {}
redshifts = {}
ras = {}
decs = {}
y0tildes = {}
xAxisSecondary = {}
yAxisSecondary = {}
nClusters2 = {} 

for x in CatalogList:
    catalogs["{0}Clusters".format(x)] = (actCat[str(SecondaryCatalogs[CatalogList.index(x)])] == 1)
    names["{0}Clusters".format(x)] = actCat[catalogs["{0}Clusters".format(x)]]['name']
    masses["{0}Clusters".format(x)] = actCat[catalogs["{0}Clusters".format(x)]]['M500cUPP']
    redshifts["{0}Clusters".format(x)] = actCat[catalogs["{0}Clusters".format(x)]]['z']
    y0tildes["{0}Clusters".format(x)] = actCat[catalogs["{0}Clusters".format(x)]]['y0tilde']
    ras["{0}Clusters".format(x)] = actCat[catalogs["{0}Clusters".format(x)]]['RADeg']
    decs["{0}Clusters".format(x)] = actCat[catalogs["{0}Clusters".format(x)]]['decDeg']
    nClusters2["{0}Clusters".format(x)] = len(masses["{0}Clusters".format(x)])
               
               
#get the toltec and cluster classes
from ClusterDash.ClusterModels.ClusterUPP import ClusterUPP
from ClusterDash.TolTEC.TolTEC import TolTEC
from ClusterDash.TolTEC.TolTEC import fwhm2sigma
from ClusterDash.DustyGalaxies.DSFGs import DSFGs
from ClusterDash.CMB.CMB import CMB

#the page header
NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("TolTEC View of ACT Clusters", className="ml-2")
                    ),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://plot.ly",
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)


#the table
TABLE = dbc.Jumbotron(
    [
        html.H1(children="Selected Cluster", className="display-7"),
        html.Hr(className="my-3"),
        dbc.Table(bordered=True, striped=True, hover=True,
                  id="ClusterTable")
    ]
)

def _update_Arrayfigure(cid_clicked, cid, checkedCatalogs, graphAxes):
    colors = ['#1f77b4']*nClusters
    
    #for setting the colors of the cluster filters
    colorsList = px.colors.qualitative.Plotly
    colors2 = {}
    for x in CatalogList:
        colors2["{0}Clusters".format(x)] = [colorsList[CatalogList.index(x)]]*nClusters2["{0}Clusters".format(x)]
    
    #set variables based on the axis
    if graphAxes == "Mass/Redshift ":
        xAxis, yAxis, xAxisSecondary, yAxisSecondary, xlabel, ylabel = 'M500cUPP','z', masses, redshifts, "M500cUPP [x 1e14]", 'Redshift'
    elif graphAxes == "RA/DEC ":
        xAxis, yAxis, xAxisSecondary, yAxisSecondary, xlabel, ylabel = 'RADeg', 'decDeg', ras, decs, 'RA (Degrees)', 'Dec (Degrees)'
        
        
    #cid_clicked is a clickData object type
    if(cid_clicked is None):
        colors[0] = '#1f77b4'
    else:
        clusterName = actCat['name'][cid]
        colors[cid] = '#ff7f0e'
        for x in CatalogList:
            if clusterName in names["{0}Clusters".format(x)]:
                point_id = names["{0}Clusters".format(x)].tolist().index(clusterName)
                colors2["{0}Clusters".format(x)][point_id] = '#ff7f0e'
                    
    #construct the figure
    fig = go.Figure()
    
    if 'ALL' in checkedCatalogs:
            fig.add_trace(go.Scatter(x=actCat[xAxis],
                             y=actCat[yAxis],
                             mode='markers',
                             marker_size=actCat['y0tilde']*10,
                             marker_color=colors,
                             opacity=0.75,
                             name="marker size prop. to y0",
                         ))

    #Traces for filtered data
    for x in CatalogList:
        if x in checkedCatalogs:
            fig.add_trace(go.Scatter(x=xAxisSecondary["{0}Clusters".format(x)],
                          y=yAxisSecondary["{0}Clusters".format(x)],
                          mode="markers",
                          marker_size=y0tildes["{0}Clusters".format(x)]*10,
                          marker_color=colors2["{0}Clusters".format(x)],
                          opacity=0.75,
                          name= x + ' Data',
                          )) 
            
    
    
    fig.update_layout(
        showlegend=True,
        hovermode="closest",
        clickmode='event+select',
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(font=dict(size=9), x=0.9, y=0.9),
        xaxis_title=xlabel,
        yaxis_title=ylabel,   
    )
    
    return fig



# a plot of mass vs. redshift or ra vs. dec with selectable points and
# filters for additional cluster catalogs
CLUSTER_PLOT = [
    dbc.CardHeader(html.H5(
        "ACT Clusters ({0:} total, {1:} low dec removed)".format(nClusters, nRemoved))),
    dbc.CardBody(
        [
            dbc.Col(dcc.RadioItems(
                id='PlotAxis',
                options=[{'label': i, 'value': i} for i in ['Mass/Redshift ', 'RA/DEC ']],
                value='Mass/Redshift ',
                inputStyle={"marginRight": "5px", "marginLeft":"20px"})),
            dbc.Col(dcc.Checklist(
                options=[
                {'label': 'All Act Clusters', 'value': 'ALL'},
                {'label': 'ACCEPT', 'value': 'ACCEPT'},
                {'label': 'Clash', 'value': 'Clash'},
                {'label': 'Herschel', 'value': 'Herschel'},
                {'label': 'NIKA', 'value': 'NIKA'},
                {'label': 'PLANCK', 'value': 'PLANCK'},
                {'label': 'RASSEBCS', 'value': 'RASSEBCS'},
                {'label': 'REFLEX', 'value': 'REFLEX'},
                {'label': 'Rosgalclus', 'value': 'Rosgalclus'},
                {'label': 'XCS', 'value': 'XCS'},
                ],
                value=['ALL'],
                inputStyle={"marginRight": "5px", "marginLeft":"20px"},
                id="catalogChecklist",),),
            dcc.Graph(id="ClusterPlot",
                      figure=_update_Arrayfigure(
                          cid_clicked=None,
                          cid = 0,
                          checkedCatalogs="ALL",
                          graphAxes='Mass/Redshift ')
            ),
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]


#a plot of the cluster y profile (UPP-based)
PROFILE_PLOT = [
    dbc.CardHeader(html.H5("PROFILE Results")),
    dbc.CardBody(
        [
            dbc.Row(dcc.Checklist(
                options=[
                    {'label': 'Show Radial Average', 'value': 'radial'},
                ],
                value=[],
                inputStyle={"marginRight": "5px", "marginLeft":"20px"},
                id="selectRadialAverage",)),
            dcc.Graph(id="ProfilePlot"),
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]


#deltaT/T spectrum
SPECTRUM_PLOT = [
    dbc.CardHeader(html.H5("SZ Spectrum")),
    dbc.CardBody(
        [
            dcc.Graph(id="SpectrumPlot"),
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]


#UPP Cluster Image (smoothed)
CLUSTER_IMAGE = [
    dbc.CardHeader(html.H5("Observed Image")),
    dbc.CardBody(
        [
        dbc.Row(
            [dbc.Col(dcc.Checklist(
                options=[
                    {'label': 'Show dusty galaxies', 'value': 'dust'},
                    {'label': 'Show array', 'value': 'array'},
                    {'label': 'Show Primary CMB', 'value': 'cmb'},
                ],
                value=['dust', 'cmb'],
                inputStyle={"marginRight": "5px", "marginLeft":"20px"},
                id="selectToShow",),),
             dbc.Col([html.I("Smoothing FWHM [Arcmin]"),
                      html.Br(),
                      dcc.Input(
                          id="input_filterFWHMArcmin",
                          type="number",
                          placeholder="None",
                          debounce=True,
                      )]),
             dbc.Col([html.I("Peculiar Velocity [km/s]"),
                      html.Br(),
                      dcc.Input(
                          id="input_peculiarVelocity",
                          type="number",
                          placeholder=0.,
                          debounce=True,
                      )]),
             dbc.Col(dcc.Dropdown(
                 id='bandDropdown',
                 options=[
                     {'label': '1.1mm Array', 'value': '1.1'},
                     {'label': '1.4mm Array', 'value': '1.4'},
                     {'label': '2.0mm Array', 'value': '2.0'}
                 ],
                 value='2.0'
             ),)
         ]),  
        dbc.Row(
            [
                dcc.Graph(id="BigClusterImage")
            ],
            style={"marginTop": 0},
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="ClusterImage2p0")),
                dbc.Col(dcc.Graph(id="ClusterImage1p4")),
                dbc.Col(dcc.Graph(id="ClusterImage1p1")),
                dbc.Col(dcc.Graph(id="DustyGalaxyImage")),
            ],
            style={"marginTop": 10},
        ),
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]



#the input boxes
c1 = [dbc.CardHeader("Input Observation Time [hrs]"),
      dbc.CardBody([
          dcc.Input(id="input_obstime", type="number",
                    placeholder="1", debounce=True),]),]

c2 = [dbc.CardHeader("Input Map Radius [arcmins]"),
      dbc.CardBody([
          dcc.Input(id="input_mapradius", type="number",
                    placeholder="3", debounce=True),]),]

c3 = [dbc.CardHeader("Atmosphere Degredation Factor"),
      dbc.CardBody([
          dcc.Slider(id="input_atmfactor", min=1, max=7, step=0.5, value=1,
                    marks={
                        1: {'label': '1', 'style': {'color': 'white'}},
                        3: {'label': '3', 'style': {'color': 'white'}},
                        5: {'label': '5', 'style': {'color': 'white'}},
                        7: {'label': '7', 'style': {'color': 'white'}}}
                     ,),]),]

OBS_PARAMETERS = [
    dbc.CardHeader(html.H5("Observation Parameters")),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card(c1, color="success", inverse=True)),
                    dbc.Col(dbc.Card(c2, color="warning", inverse=True)),
                    dbc.Col(dbc.Card(c3, color="danger", inverse=True)),
                ],
                className="mb-4",
            ),
        ],
        style={"marginTop": 0, "marginBottom": 0},
    )
]


#this sets the layout of the entire page using the sections defined
#above
BODY = dbc.Container(
    [
        dcc.Store(id="browser_json_data"),
        dcc.Store(id="browser_clickdata"),
        dbc.Row(
            [
                dbc.Col(TABLE, width=4, align="top"),
                dbc.Col(dbc.Card(CLUSTER_PLOT), width=8),
            ],
            style={"marginTop": 10},
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Card(OBS_PARAMETERS)),
            ],
            style={"marginTop": 0},
        ),
        dbc.Row(
            [
                dbc.Col([dbc.Card(PROFILE_PLOT),
                        dbc.Card(SPECTRUM_PLOT)],
                        width=6),
                dbc.Col(dbc.Card(CLUSTER_IMAGE), width=6),
            ],
            style={"marginTop": 10},
        ),
    ],
    className="mt-12",
    fluid=True,
)
app.layout = html.Div(children=[NAVBAR, BODY])



# CALLBACKS
#callback that identifies which ACT cluster was clicked on or returns 0
@app.callback(
    Output("browser_json_data", "data"),
    [
        Input('ClusterPlot', 'clickData'),
    ],
    State('PlotAxis', 'value')
)
def holdOnCluster(cid_clicked, graphAxes):
    if graphAxes == "Mass/Redshift ":
        xAxis, yAxis = 'M500cUPP','z'
    elif graphAxes == "RA/DEC ":
        xAxis, yAxis = 'RADeg', 'decDeg'

    print()
    print(cid_clicked)
    print()
    
    if(cid_clicked is None):
        cid = 0
    else:
        cid = np.where((actCat[yAxis] == cid_clicked['points'][0]['y']) &
                       (actCat[xAxis] == cid_clicked['points'][0]['x']))[0]
        if(len(cid) == 0):
            return 0
        else:
            cid = int(cid[0])
    
    return cid


#callback that identifies which ACT cluster was clicked on or returns 0
@app.callback(
    Output("browser_clickdata", "data"),
    [
        Input('ClusterPlot', 'clickData')
    ]
)
def saveClickdata(cid_clicked):
    return cid_clicked


#TABLE
@app.callback(Output("ClusterTable","children"),[Input("browser_json_data", "data")])
def update_table(cid):
    name = actCat[cid]['name']
    ra = actCat[cid]['RADeg']
    dec = actCat[cid]['decDeg']
    y0 = actCat[cid]['y0tilde']*1.e-4
    y0err = actCat[cid]['y0tilde_err']*1.e-4
    z = actCat[cid]['z']
    M500 = actCat[cid]['M500cUPP']*1.e14
    
    #Determine what ancillary data is available for it
    ancildat = []
    
    for i in range(len(CatalogList)):
        if actCat[cid][SecondaryCatalogs[i]]==1:
            ancildat.append(CatalogList[i] + ", ")
    
    if len(ancildat) == 0:
        ancildat = "None"
            

    #form the table
    bod = []
    bod.append(html.Tr([html.Td("Name"), html.Td(name)], className ='table-success'))
    bod.append(html.Tr([html.Td("RA, Dec [deg]"), html.Td("{0:3.3f}, {1:3.3f}".format(ra,dec))]))
    bod.append(html.Tr([html.Td("y0"), html.Td("{0:2.2e} +/- {1:2.2e}".format(y0,y0err))]))
    bod.append(html.Tr([html.Td("z"), html.Td("{0:2.2f}".format(z))]))
    bod.append(html.Tr([html.Td("M_{500}"), html.Td("{0:2.2e} M_sun".format(M500))]))
    bod.append(html.Tr([html.Td("Additional Data Available"), html.Td(ancildat)]))
    
    table_body = [html.Tbody(bod)]
    return table_body


#CLUSTER_PLOT
@app.callback(
    Output("ClusterPlot","figure"),
    [
        Input("browser_clickdata", "data"),
        Input("browser_json_data", "data"),
        Input("catalogChecklist", "value"),
        Input("PlotAxis", "value")
    ]
)
def update_Arrayfigure(cid_clicked, cid, checkedCatalogs, graphAxes):
    return _update_Arrayfigure(cid_clicked, cid, checkedCatalogs, graphAxes)


#PROFILE_PLOT
@app.callback(Output("ProfilePlot","figure"),
              [Input("browser_json_data", "data"),
               Input("input_obstime", "value"),
               Input("input_mapradius", "value"),
               Input("input_atmfactor", "value"),
               Input("selectRadialAverage", "value"),
               Input("input_peculiarVelocity", "value"),
              ]
)
def updateProfileFigure(cid, time, radius, atmFactor, radAvg, velocity):
    if(time == None):
        time = 1.
    if(radius == None):
        radius = 3.
    if(atmFactor == None):
        atmFactor = 1.
    if(velocity == None):
        velocity=0.

    #more bounds checking
    time = max(time, 0.1)
    radius = max(radius, 3.)

    #should we show the radial average instead?
    showRadialAverage = radAvg.count('radial')

    c = getCluster(cid, velocity=velocity)
    npts=100
    thetaArcmin = np.linspace(0.,1.2*radius,npts)
    y = c.y(thetaArcmin)
    s = np.r_[np.flip(y)[0:npts-1],y]
    t = np.r_[-np.flip(thetaArcmin)[0:npts-1],thetaArcmin]
    sigmaArcmin = (10./60.)/2.35482004503
    g = np.exp(-0.5*(t/sigmaArcmin)**2)
    g = g[npts-11:npts+10]
    s = np.convolve(s, g/g.sum(), mode="valid")  
    w = np.argmax(s)
    s = s[w:]
    t = thetaArcmin[0:len(s)]
    obsy = interpol.interp1d(t,s,kind='quadratic')
    
    fig = go.Figure()
    # Set axes ranges
    fig.update_xaxes(range=[0, radius])
    fig.update_yaxes(range=[0.01, y.max()*1.e4])
    fig.update_layout(showlegend=True)

    # the actual profile
    fig.add_trace(go.Scatter(x=thetaArcmin, y=y*1.e4,
                             mode='lines', name="UPP generated y",
                             ))    
    
    # the observed profile
    fig.add_trace(go.Scatter(x=t, y=s*1.e4,
                             mode='lines', name="convolved with TolTEC beam",))
    
    # R500 line
    fig.add_shape(
        dict(type="line",
             x0=c.theta500arcmin, y0=0.01,
             x1=c.theta500arcmin, y1=y.max()*0.95e4,
             name="R500",
             line=dict(color="crimson",width=2,dash="dashdot",)))

    fig.update_layout(
        xaxis=dict(
            title="Theta Obs [arcmin]",
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            title="Compton y [x 1.e4]",
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
        ),
        autosize=True,
        margin=dict(autoexpand=False,l=100,r=20, t=20,),
        showlegend=True,
        plot_bgcolor='white',
        annotations = [dict(
            x=c.theta500arcmin,
            y=y.max()*1.e4,
            text="R500",
            xref="x", yref="y",
            showarrow=False,
            font=dict(
            family="Courier New, monospace",
            size=16,
            color="crimson"
            ),)]
    )

    #put in the error bars for a TolTEC observation here
    obsAreaArcmin2 = 4.*radius**2
    obsTime = time
    T = TolTEC(2.0, atmFactor=atmFactor)
    depth = T.depth_y(obsAreaArcmin2, obsTime)*1.e4
    t10 = np.arange(0.,radius,1./6.)
    y10 = obsy(t10)*1.e4
    errorDict = dict(
        type='data', array=[depth]*len(y10), visible=True,)
    fig.add_trace(go.Scatter(x=t10,
                             y=y10,
                             mode='markers',
                             error_y=errorDict))    

    if(showRadialAverage):
        #need to calculate the radial average points
        radArcsec = t10*60.
        binwidth = 10.
        circumference = 2.*np.pi*radArcsec
        points = circumference/binwidth
        points[0] = 1.
        depth = np.array([depth]*len(t10))/np.sqrt(points)
        errorDict = dict(
            type='data', array=depth, visible=True,)
        fig.add_trace(go.Scatter(x=t10,
                                 y=y10,
                                 mode='markers',
                                 error_y=errorDict))


    return fig



@cache_func
def makeCMBRealization(nPixX, nPixY, pixSizeArcsec):
    cmb = CMB(nPixX, nPixY, pixSizeArcsec)
    return cmb.CMB_T*1.e-6


@cache_func
def makeDustyGalaxies(mapAreaDeg2, nPixX, nPixY):
    d = DSFGs(mapAreaDeg2)
    ux, uy = d.pointSourceMapCoords(nPixX, nPixY)
    sources = dict(ux=ux, uy=uy, S1p1=d.S1p1, S1p4=d.S1p4, S2p0=d.S2p0)
    return sources


@cache_func
def getImageParameters(radiusArcmin):
    if(radiusArcmin == None):
        radiusArcmin = 3.
    radiusArcmin = max(radiusArcmin, 3.)
    nPixX = int(np.ceil(radiusArcmin*60*2))
    nPixY = int(np.ceil(radiusArcmin*60*2))
    x = np.linspace(-radiusArcmin, radiusArcmin, nPixX)
    y = np.linspace(-radiusArcmin, radiusArcmin, nPixY)
    pixSizeArcsec = (x[1]-x[0])*60.
    mapAreaDeg2 = 4.*radiusArcmin**2/3600.
    imageParams = dict(
        radiusArcmin=radiusArcmin,
        nPixX = nPixX,
        nPixY = nPixY,
        xArcmin = x,
        yArcmin = y,
        pixSizeArcsec = pixSizeArcsec,
        mapAreaDeg2 = mapAreaDeg2,)
    return imageParams


@cache_func
def getCluster(cid, velocity=0.):
    if(cid is None):
        cid = 0
    if(velocity == None):
        velocity=0.
    M500 = actCat[cid]['M500cUPP']*1.e14
    z = actCat[cid]['z']
    c = ClusterUPP(M500,z,velocity=velocity)
    return c


@cache_func
def getRawClusterImages(c, radiusArcmin):
    ySZ, kSZ, x, y = c.SZImages(radiusArcmin) 
    return ySZ, kSZ, x, y
    

def getTolTECmap(band, cluster, time, radius, atmFactor, showDust=1, showCMB=1, smoothingFWHM=0):
    if(time == None):
        time = 1.
    if(radius == None):
        radius = 3.
    if(atmFactor == None):
        atmFactor = 1.
    time = max(time, 0.1)
    radius = max(radius, 3.)

    #fetch image parameters
    iP = getImageParameters(radius)
    pixSizeArcsec = iP["pixSizeArcsec"]
    nPixX = iP["nPixX"]
    nPixY = iP["nPixY"]
    
    #make a TolTEC
    T = TolTEC(band, atmFactor=atmFactor)
    shapes = T.makeArrayShapes()
    
    #start with the cluster
    cimg, kimg, x, y = getRawClusterImages(cluster, radius)
    cimg = T.y2K(cimg)
    cimg = cimg + kimg    
    cimg = ndimage.gaussian_filter(cimg, fwhm2sigma(T.fwhm)/pixSizeArcsec)
    
    #add the dusty galaxies
    dG = makeDustyGalaxies(iP["mapAreaDeg2"], nPixX, nPixY)
    if(band==1.1):
        SmJy = np.array(dG['S1p1'])
    elif(band==1.4):
        SmJy = np.array(dG['S1p4'])
    else:
        SmJy = np.array(dG['S2p0'])
    dimg = np.zeros((nPixX,nPixY), dtype="float")
    dimg[dG["ux"],dG["uy"]] = SmJy
    dimg = ndimage.gaussian_filter(dimg, fwhm2sigma(T.fwhm)/pixSizeArcsec)
    dimg *= SmJy.max()/dimg.max()
    dimg = T.Jy2K(dimg*1.e-3)
    if(showDust == 0):
        dimg *= 0.
    dimg += cimg

    #add in the CMB (no need to smooth) if requested
    if(showCMB):
        dimg += makeCMBRealization(nPixX, nPixY, pixSizeArcsec)
        
    #put in the noise from a TolTEC observation here
    noise_K = np.zeros((nPixX,nPixY),dtype="float")
    obsAreaDeg2 = iP["mapAreaDeg2"]
    obsTime = time
    depth_mJy = T.depth_mJy(obsAreaDeg2, obsTime)
    depth_K = T.Jy2K(depth_mJy*1.e-3)
    noise_K = T.Jy2K(T.noiseMap_mJy(nPixX, nPixY, pixSizeArcsec, depth_mJy)*1.e-3)
    dimg += noise_K

    #do some extra smoothing if requested
    if(smoothingFWHM > 0):
        dimg = ndimage.gaussian_filter(dimg, fwhm2sigma(smoothingFWHM*60.)/pixSizeArcsec)
    
    mapDepths = (depth_mJy,depth_K)
    image = (dimg, x, y)

    return image, shapes, mapDepths


def getDustyGalaxyImage(band, radius):
    if(radius == None):
        radius = 3.
    radius = max(radius, 3.)

    #fetch image parameters
    iP = getImageParameters(radius)
    pixSizeArcsec = iP["pixSizeArcsec"]
    nPixX = iP["nPixX"]
    nPixY = iP["nPixY"]
    x = iP["xArcmin"]
    y = iP["yArcmin"]
    
    #make a TolTEC
    T = TolTEC(band)

    #add the dusty galaxies
    dG = makeDustyGalaxies(iP["mapAreaDeg2"], nPixX, nPixY)
    if(band==1.1):
        SmJy = np.array(dG['S1p1'])
    elif(band==1.4):
        SmJy = np.array(dG['S1p4'])
    else:
        SmJy = np.array(dG['S2p0'])
    dimg = np.zeros((nPixX,nPixY), dtype="float")
    dimg[dG["ux"],dG["uy"]] = SmJy
    
    #convolve
    dimg = ndimage.gaussian_filter(dimg, fwhm2sigma(T.fwhm)/pixSizeArcsec)
    dimg = dimg/dimg.max()*SmJy.max()
    return dimg, x, y


def generateImage(z,x,y,title,width,height,cbTitle="deltaT [uK]"):
    fig = go.Figure()    
    #generate the figure
    fig.add_trace(
        go.Heatmap(
            z=z, x=x, y=y,
            colorbar=dict(
                title=cbTitle,
                titleside="top",
                ticks="outside",
            ), 
        ),
    )
    t = {
        'text': title,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
    
    fig.update_layout(
        title=t,
        autosize=True,
        width=width,
        height=height,
        xaxis_title="Theta Obs [arcmin]",
        yaxis_title="Theta Obs [arcmin]")

    return fig


#IMAGES
@app.callback(Output("ClusterImage2p0","figure"),
              [Input("browser_json_data", "data"),
               Input("input_obstime", "value"),
               Input("input_mapradius", "value"),
               Input("input_atmfactor", "value"),
               Input("input_peculiarVelocity", "value"),
               ])
def updateClusterImage2p0(cid, time, radius, atmFactor, velocity):
    if(velocity == None):
        velocity=0.
    
    cluster = getCluster(cid, velocity=velocity)
    
    #get TolTEC maps
    image, shapes, depths = getTolTECmap(2.0, cluster, time, radius, atmFactor)
    img2p0 = image[0]
    x = image[1]
    y = image[2]
    fig = generateImage(img2p0*1.e6, x, y, "2.0mm Array", 400, 400)    
    return fig


@app.callback(Output("ClusterImage1p4","figure"),
              [Input("browser_json_data", "data"),
               Input("input_obstime", "value"),
               Input("input_mapradius", "value"),
               Input("input_atmfactor", "value"),
               Input("input_peculiarVelocity", "value"),
               ])
def updateClusterImage1p4(cid, time, radius, atmFactor, velocity):
    if(velocity == None):
        velocity=0.
        
    cluster = getCluster(cid, velocity)
        
    #get TolTEC maps
    image, shapes, depths = getTolTECmap(1.4, cluster, time, radius, atmFactor)
    img1p4 = image[0]
    x = image[1]
    y = image[2]
    fig = generateImage(img1p4*1.e6, x, y, "1.4mm Array", 400, 400)
    return fig


@app.callback(Output("ClusterImage1p1","figure"),
              [Input("browser_json_data", "data"),
               Input("input_obstime", "value"),
               Input("input_mapradius", "value"),
               Input("input_atmfactor", "value"),
               Input("input_peculiarVelocity", "value"),
               ])
def updateClusterImage1p1(cid, time, radius, atmFactor, velocity):
    if(velocity == None):
        velocity=0.
    
    cluster = getCluster(cid, velocity=velocity)
        
    #get TolTEC maps
    image, shapes, depths = getTolTECmap(1.1, cluster, time, radius, atmFactor)
    img1p1 = image[0]
    x = image[1]
    y = image[2]
    fig = generateImage(img1p1*1.e3, x, y, "1.1mm Array", 400, 400, cbTitle="mK")
    return fig




@app.callback(Output("DustyGalaxyImage","figure"),
              [Input("input_mapradius", "value"),
           ])
def updateDustyGalaxyImage(radius):
    #get dusty galaxy map
    img,x,y = getDustyGalaxyImage(1.1, radius)
    fig = generateImage(img, x, y, "Dusty Galaxies", 400, 400,
                        cbTitle="mJy")
    return fig


@app.callback(Output("BigClusterImage","figure"),
              [Input("browser_json_data", "data"),
               Input("input_obstime", "value"),
               Input("input_mapradius", "value"),
               Input("input_atmfactor", "value"),
               Input("selectToShow", "value"),
               Input("bandDropdown", "value"),
               Input("input_filterFWHMArcmin", "value"),
               Input("input_peculiarVelocity", "value"),
               ])
def updateBigClusterImage(cid, time, radius, atmFactor, s2s, band, smoothingFWHM, velocity):
    if(velocity == None):
        velocity=0.
    if(smoothingFWHM == None):
        smoothingFWHM = 0.
    showDust = s2s.count('dust')
    showArray = s2s.count('array')
    showCMB = s2s.count('cmb')
    if(band == "1.1"):
        b = 1.1
        title = "1.1mm Array Image"
    elif(band == "1.4"):
        b = 1.4
        title = "1.4mm Array Image"
    else:
        b = 2.0
        title = "2.0mm Array Image"
    
    cluster = getCluster(cid, velocity=velocity)
        
    #get TolTEC maps
    image, shapes, depths = getTolTECmap(b, cluster, time, radius, atmFactor,
                                         showDust=showDust, showCMB=showCMB,
                                         smoothingFWHM=smoothingFWHM)
    img = image[0]
    x = image[1]
    y = image[2]
    depth_mJy = depths[0]
    depth_K = depths[1]

    #update the title
    title += " (rms={0:3.1f}uJy, {1:3.1f}uK)".format(depth_mJy*1.e3, depth_K*1.e6)
    
    fig = generateImage(img*1.e6, x, y, title, 800, 800)

    #overplot array shapes if requested
    if(showArray):
        fig.update_layout(shapes=shapes)
    return fig



#SPECTRUM_PLOT
@app.callback(Output("SpectrumPlot","figure"),
              [Input("browser_json_data", "data"),
               Input("input_peculiarVelocity", "value"),
               ])
def updateSpectrumFigure(cid, velocity):
    if(velocity == None):
        velocity=0.
        
    cluster = getCluster(cid, velocity)
    y0 = cluster.y(0.)
    
    #for the plot x axis
    nu_GHz = np.linspace(30.,500.,100)
    dI = cluster.deltaI(y0,nu_GHz)
    
    fig = go.Figure()

    # the actual profile
    fig.add_trace(go.Scatter(x=nu_GHz, y=dI,
                             mode='lines', name="cluster SZ spectrum",))

    #add the TolTEC bands
    t2x = [128,170]
    t1p4x = [195,245]
    t1p1x = [245,310]
    fig.add_shape(type="rect",
                  xref="x", x0=t2x[0], x1=t2x[1],
                  yref="paper", y0=0, y1=1,
                  fillcolor="LightSkyBlue",
                  opacity=0.2)
    fig.add_shape(type="rect",
                  xref="x", x0=t1p4x[0], x1=t1p4x[1],
                  yref="paper", y0=0, y1=1,
                  fillcolor="LightSkyBlue",
                  opacity=0.2)
    fig.add_shape(type="rect",
                  xref="x", x0=t1p1x[0], x1=t1p1x[1],
                  yref="paper", y0=0, y1=1,
                  fillcolor="LightSkyBlue",
                  opacity=0.2)
    
    fig.update_layout(
        xaxis=dict(
            title="Frequency [GHz]",
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            title="delta I [MJy/sr]",
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
        ),
        autosize=True,
        margin=dict(autoexpand=False,l=100,r=20, t=20,),
        showlegend=False,
        plot_bgcolor='white',
    )

    return fig


# Main
if __name__ == "__main__":
    app.run_server(debug=True)
