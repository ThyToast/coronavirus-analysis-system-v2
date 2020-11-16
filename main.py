import re
from time import sleep

import altair as alt
import covid_daily
import numpy as np
import pandas as pd
import streamlit as st
import tweepy
from googletrans import Translator
from statsmodels.tsa.arima_model import ARIMA
from textblob import TextBlob

from covid_bot import covid_bot


# cached function for fast response
@st.cache(show_spinner=False)
def getData():
    with st.spinner(text="Fetching data..."):
        url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
        df = pd.DataFrame(data=pd.read_csv(url))
        return df


@st.cache(show_spinner=False)
def getReport(country: str):
    with st.spinner(text="Fetching data..."):
        if country == 'United States':
            country = 'USA'
        report = covid_daily.overview(as_json=False)
        report = report[report['Country,Other'] == country]
        return report


def cleanText(text):
    text = re.sub('@[A-Za-z0–9]+:', '', text, flags=re.MULTILINE)
    text = re.sub(r"(?:\@|https?\://)\S+", "", text, flags=re.MULTILINE)
    text = re.sub('#', '', text, flags=re.MULTILINE)
    # text = re.sub('\+n', '', text, flags=re.MULTILINE)
    # text = re.sub('\n', '', text, flags=re.MULTILINE)
    text = re.sub('RT[\s]+', '', text, flags=re.MULTILINE)
    text = re.sub('⃣', '', text, flags=re.MULTILINE)
    text = re.sub('&amp;', '', text, flags=re.MULTILINE)
    # text = re.sub(' +', ' ', text, flags=re.MULTILINE)
    return text


@st.cache(show_spinner=False)
def getTranslate(text):
    translator = Translator()
    result = None
    while result is None:
        try:
            result = translator.translate(text).text
        except Exception as e:
            print(e)
            translator = Translator()
            sleep(0.5)
            pass
    return result


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def getPolarity(text):
    return TextBlob(text).sentiment.polarity


def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


@st.cache(show_spinner=False)
def getTwitterData(userName: str):
    with st.spinner(text="Fetching data..."):
        consumerKey = '2GEG6e2BlCA79Iw1BDZTMcfsm'
        consumerSecret = 'KbqBsUxLWEhyDCWwUQ5rEyCRB2DDq3MtUsLrpi4WRmMqRmaZ7e'
        accessToken = '1862224740-JBq4GzpKSYVbWnwWvtU2EcxocA9TNYqjF0SWsed'
        accessSecret = 'UzTpWyApEQ5UpJIXnVeIPWV8sLoPvnKmDOtzfT9hpOMmO'

        authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
        authenticate.set_access_token(accessToken, accessSecret)
        api = tweepy.API(authenticate, wait_on_rate_limit=True)

        posts = tweepy.Cursor(api.search, q="COVID-19 from:" + userName, rpp=100, tweet_mode="extended").items()

        # posts = api.user_timeline(screen_name=userName, count=100, lang="en", tweet_mode="extended")
        df_twitter = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
        df_twitter = df_twitter.head(20)

        # Data cleaning & translation
        df_twitter['Tweets'] = df_twitter['Tweets'].apply(cleanText)
        df_twitter['Tweets'] = df_twitter['Tweets'].apply(getTranslate)
        df_twitter['Subjectivity'] = df_twitter['Tweets'].apply(getSubjectivity)
        df_twitter['Polarity'] = df_twitter['Tweets'].apply(getPolarity)
        df_twitter['Analysis'] = df_twitter['Polarity'].apply(getAnalysis)

        return df_twitter


def forecastDf(df, country: str, index: int):
    y = df['Total Cases']
    y = y.astype(np.int)
    y = np.asanyarray(y)

    df['Case Type'] = "Actual Cases"
    df['Date'] = pd.to_datetime(df['Date'])

    model_ar_confirmed = ARIMA(y, order=(2, 0, 0))
    model_fit_ar_confirmed = model_ar_confirmed.fit(disp=False)
    predict_ar_confirmed = model_fit_ar_confirmed.predict(1, (len(y) + index - 1))

    ftr = df.append(pd.DataFrame({'Date': pd.date_range(start=df['Date'].iloc[-1], periods=index, freq='d',
                                                        closed='right')}))
    ftr['Total Cases'] = predict_ar_confirmed
    ftr['Case Type'] = "Forecasted Cases"
    ftr = ftr.reset_index(drop=True)

    df = df.append(ftr)
    chart = alt.Chart(df).mark_line().encode(
        y='Total Cases:Q',
        x='Date:T',
        color='Case Type:N',
        tooltip=['Date', 'Case Type', 'Total Cases']
    ).properties(
        width=700,
        height=500
    ).interactive()
    st.altair_chart(chart)
    st.write(ftr.sort_values(by=['Date'], ascending=False))

    # plt.plot(y, label='Actual Data', color='blue')
    # plt.plot(predict_ar_confirmed, label='Forecasted unknown data (Future)', color='orange')
    # plt.plot(predict_ar_confirmed[:len(
    #     predict_ar_confirmed)-64], label='Forecasted known data (Past/Present)', color='red')
    #
    # plt.title('COVID-19 Prediction for ' + country)
    # plt.xlabel('Time (Days)')
    # plt.ylabel('No. of Infected')
    # plt.legend()
    # plt.show()


st.set_page_config(
    layout="centered",
    initial_sidebar_state="auto",
    page_title="Viral Infection Analysis System",  # String or None. Strings get appended with "• Streamlit".
    page_icon="coronavirus.ico",
)

dark_theme = st.sidebar.checkbox("Dark theme")
if dark_theme:
    # CSS Theme
    st.markdown(
        """
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans&display=swap" rel="stylesheet">
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#303039,#303039);
        color: #303039;
    }
    .Widget>label {
        color: white;
        background-color: #303039;
    }
    .sidebar .sidebar-content h1{
        color: white;
    }
     .reportview-container li{
        font-size: large;
    }
    .reportview-container h3{
        font-size: large;
    }
    .reportview-container dl, .reportview-container ol, .reportview-container p, .reportview-container ul{
        font-size: large;
    }
    .Widget>label{
        font-size: medium;
    }
    [class^="st-b"]  {
        color: white;
    }
    [class^="st-ae st-fh st-fi st-fj st-c5 st-fk st-fa st-fl st-fm"]  {
        color: white;
    }
    [class^="st-ae st-af st-ag st-ah st-fn st-f8 st-fl st-fo st-fp"]  {
        color: white !important;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-cz {
        fill: white;
    }
    .st-dr{
        fill: white;
    }
    .st-ae st-af st-ag st-ah st-fn st-f8 st-fl st-fo st-fp{
        color: white !important;
    }
    .st-fn{
        color: white !important;
    }
    .st-bm
    {
        color: white !important;
    }
     .st-ck{
        color: white !important;
    }
    .st-co{
        color: white !important;
    }
    .st-bn{
        color: white;
    }
    .btn-outline-secondary{
        border-color: #e83e8c;
        color: #e83e8c;
    }
    .st-at {
        background-color: #303039;
    }
    .st-df {
        background-color: #303039;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

countries = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Anguilla', 'Antigua and Barbuda', 'Argentina',
             'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados',
             'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bonaire Sint Eustatius and Saba',
             'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'British Virgin Islands', 'Brunei', 'Bulgaria',
             'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands',
             'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Congo', 'Costa Rica', "Cote d'Ivoire",
             'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic', 'Democratic Republic of Congo', 'Denmark',
             'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea',
             'Eritrea', 'Estonia', 'Ethiopia', 'Faeroe Islands', 'Falkland Islands', 'Fiji', 'Finland', 'France',
             'French Polynesia', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland',
             'Grenada', 'Guam', 'Guatemala', 'Guernsey', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras',
             'Hungary', 'Iceland', 'India', 'Indonesia', 'International', 'Iran', 'Iraq', 'Ireland', 'Isle of Man',
             'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait',
             'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg',
             'Macedonia', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Mauritania', 'Mauritius',
             'Mexico', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Montserrat', 'Morocco', 'Mozambique', 'Myanmar',
             'Namibia', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria',
             'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palestine', 'Panama', 'Papua New Guinea',
             'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Romania', 'Russia',
             'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'San Marino',
             'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore',
             'Sint Maarten (Dutch part)', 'Slovakia', 'Slovenia', 'Somalia', 'South Africa', 'South Korea',
             'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria',
             'Taiwan', 'Tanzania', 'Thailand', 'Timor', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
             'Turks and Caicos Islands', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States',
             'Uruguay', 'Uzbekistan', 'Vatican', 'Venezuela', 'Vietnam', 'Zambia',
             'Zimbabwe']

st.title("Viral Infection Analysis System 🦠😷")
st.sidebar.title("Menu")
page_select = st.sidebar.radio("Select page to view", ('Overview', 'COVID-19 Cases', 'COVID-19 Forecast', 'COVID-Bot',
                                                       'Health '
                                                       'Advice & '
                                                       'Report'))
# when overview is selected
if page_select == 'Overview':
    st.write("## Overview \n > - The Coronavirus disease 2019 (COVID-19) is defined as illness caused by a novel "
             "coronavirus now called severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) which was first "
             "identified amid an outbreak of respiratory illness cases in Wuhan City, "
             "Hubei Province, China. \n > - As of November 2020, there are **51 million** total cases worldwide and "
             "**1.2 million** dead from the Coronavirus alone.")
    st.write("\n")
    st.write("## The Idea")
    st.write("> Introducing the Viral Infection Analysis System or ViralNet for short, it aims to combine web scraping "
             "as well as data forecasting to visualize pandemics and inform the public with the latest news in an "
             "convenient and interactive dashboard. \n\n > Feel free to explore the app on the menu to your left.")
    st.write("\n ## Our goal is to: ")
    col1, col2, col3 = st.beta_columns(3)
    col1.write("- Display the latest health news and reports")
    col1.image('images/system.png', width=150)
    col2.write("- Create a monitoring system for pandemics like COVID-19")
    col2.image('images/strategic-plan.png', width=150)
    col3.write("- Benchmark the actions taken by governments towards pandemics")
    col3.image('images/benchmark.png', width=150)

# when health advice is selected
if page_select == 'Health Advice & Report':
    st.write("> This page consists of news and reports regarding COVID-19 extracted from Twitter, you may select the "
             "data source from the menu")

    twitter_user = st.sidebar.selectbox("Select source of health advice and report", ('Ministry of Health Malaysia',
                                                                                      'World Health Organisation',
                                                                                      'Malaysiakini',
                                                                                      'The Star Online'))
    st.write(" ## News & reports from the " + twitter_user)

    if twitter_user == 'Ministry of Health Malaysia':
        twitter_user = 'KKMPutrajaya'
    if twitter_user == 'World Health Organisation':
        twitter_user = 'WHO'
    if twitter_user == 'Malaysiakini':
        twitter_user = 'malaysiakini'
    if twitter_user == 'The Star Online':
        twitter_user = 'staronline'

    posts = getTwitterData(twitter_user)
    # st.write(posts)
    j = 1
    sortedDF = posts.sort_values(by=['Polarity'])
    # for i in range(0, sortedDF.shape[0]):
    #     if sortedDF['Analysis'][i] == 'Positive':
    #         st.write('### ** ' + str(j) + ')  **' + sortedDF['Tweets'][i])
    #         j = j + 1

    for i in range(0, sortedDF.shape[0]):
        st.write('### ** ' + str(j) + ')  **' + sortedDF['Tweets'][i])
        j = j + 1

# when covid 19 cases is selected
if page_select == "COVID-19 Cases":
    country_name = st.sidebar.selectbox("Select countries 🌎", countries)
    st.write("> This page will display the latest information about COVID-19 cases worldwide provided by *Our "
             "World in Data* and *Worldometers*")

    data = getData()
    df = data[data['location'] == country_name].iloc[:, 2:9].sort_values(by=['date', 'total_cases'], ascending=False). \
        replace(np.nan, 0) \
        .drop(['location', 'new_cases_smoothed'], axis=1).reset_index(drop=True)
    df['date'] = df['date'].astype('datetime64[ns]')
    df.columns = ['Date', 'Total Cases', 'New Cases', 'Total Deaths', 'New Deaths']
    overview = getReport(country_name)
    st.write("> You may browse the countries available in the menu and choose what "
             "statistics you would like to see in the table below")
    st.write(" ## COVID-19 reports in " + country_name + " as of " + df['Date'][0].strftime("%d %B, %Y"))

    # st.write("### **Current reports: ** \n (source: [worldometers.info]("
    #          "https://www.worldometers.info/coronavirus/))")
    st.write("- ### " + overview['NewCases'].values[0].astype(int).astype(str) + " new cases \n- ### " +
             overview['ActiveCases'].values[0].astype(int).astype(str) + " active cases \n- ### " +
             overview['TotalCases'].values[0].astype(int).astype(str) + " infected in total\n- ### " +
             overview['NewDeaths'].values[0].astype(int).astype(str) + " new deaths\n- ### " +
             overview['TotalDeaths'].values[0].astype(int).astype(str) + " deaths in total")

    chart_data = df.set_index("Date")

    cases_type = st.multiselect("Click below to select data",
                                ("New Cases", "Total Cases", "New Deaths", "Total Deaths"),
                                ["New Cases"])
    data = df.melt('Date', var_name='Case Type', value_name='Number of Cases')
    available = data['Case Type'].isin(cases_type)
    data = data[available]

    if not cases_type:
        chart = st.empty
        st.error("Please select at least one data.")

    if cases_type:
        chart = alt.Chart(data).mark_bar().encode(
            y='Number of Cases:Q',
            x='Date:T',
            color='Case Type:N',
            tooltip=['Date', 'Case Type', 'Number of Cases']
        ).properties(
            width=700,
            height=500
        ).interactive()
        st.altair_chart(chart)

    df = df.set_index("Date")
    st.write(df)

# when display forecast is selected
if page_select == "COVID-19 Forecast":
    country_name = st.sidebar.selectbox("Select countries 🌎", countries)

    st.write("> This page will forecast the trend of COVID-19 cases using the ARIMA model, you may select the "
             "countries on the menu and use the slider to adjust forecast length")

    st.write(" ## Here are the forecasts for COVID-19 in " + country_name)
    data = getData()
    df = data[data['location'] == country_name].iloc[:, 2:5].sort_values(by=['date'], ascending=True). \
        replace(np.nan, 0) \
        .drop(['location'], axis=1).reset_index(drop=True)
    df['date'] = df['date'].astype('datetime64[ns]')
    df.columns = ['Date', 'Total Cases']

    index = st.slider('Select how many days to forecast : ', 1, 365)
    forecastDf(df, country_name, index)

if page_select == 'COVID-Bot':
    st.write("> COVID-BOT is a NLP bot trained with basic COVID-19 corpus using CNN architecture, it will answer "
             "basic questions and provide facts about COVID-19")
    user_input = covid_bot.CovidBot.get_text()
    response = covid_bot.CovidBot.botResponse(user_input)
    bot = st.text_area("Bot:", value=response, height=200, max_chars=None, key=None)
    if bot:
        button = st.checkbox("Click for voiced chatbot")
        if button:
            covid_bot.CovidBot.speak(response)


# when covid 19 cases is selected (data based on worldometers)
# if page_select == "COVID-19 Cases":
#     countries = AVAILABLE_COUNTRIES
#     countries = [country.capitalize() for country in countries]
#     country_name = st.sidebar.selectbox("Select countries 🌎", countries)
#     st.write(" ## Here are the latest cases of COVID-19 in " + country_name)
#
#     df = getData(country_name)
#     df = df.sort_values(by=['Date', 'Total Cases'], ascending=False). \
#         replace(np.nan, 0).rename(columns = {'Novel Coronavirus Daily Cases': 'New Cases'})
#
#     overview = getReport(country_name)
#
#     day = df.index[0]
#     day += datetime.timedelta(days=1)
#     day = day.strftime("%d %B, %Y")
#
#     st.write("### **Current reports: ** \n (source: [worldometers.info]("
#              "https://www.worldometers.info/coronavirus/))")
#     st.write(" ### Updated as of " + day)
#     st.write("- ### " + overview['NewCases'].values[0].astype(int).astype(str) + " new cases \n- ### " +
#              overview['ActiveCases'].values[0].astype(int).astype(str) + " active cases \n- ### " +
#              overview['TotalCases'].values[0].astype(int).astype(str) + " infected in total\n- ### " +
#              overview['NewDeaths'].values[0].astype(int).astype(str) + " new deaths\n- ### " +
#              overview['TotalDeaths'].values[0].astype(int).astype(str) + " deaths in total")
#
#     cases_type = st.multiselect("Select data to show", ("New Cases", "Total Cases", "New Deaths", "Total Deaths"),
#                                 ["New Cases"])
#     df.reset_index(level=0, inplace=True)
#     data = df.melt('Date', var_name='Case Type', value_name='Number of Cases')
#     available = data['Case Type'].isin(cases_type)
#     data = data[available]
#
#     if not cases_type:
#         chart = st.empty
#         st.error("Please select at least one data.")
#
#     if cases_type:
#         chart = alt.Chart(data).mark_bar().encode(
#             y='Number of Cases:Q',
#             x='Date:T',
#             color='Case Type:N',
#             tooltip=['Date', 'Case Type', 'Number of Cases']
#         ).properties(
#             width=700,
#             height=500
#         ).interactive()
#         st.altair_chart(chart)
#
#     df = df.set_index("Date")
#     st.write(df)
