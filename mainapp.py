#%%
import streamlit as st
# set light/dark theme
st.set_page_config(layout="wide", page_title="Imprecision", page_icon=":moneybag:", initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
from scipy.stats import norm
#import mpld3
#import streamlit.components.v1 as components



nu_slider=st.sidebar.slider("nu", 0.01, 0.5, 0.07, 0.01)
sigma_slider=st.sidebar.slider("sigma", 0.01, 0.5, 0.26, 0.01)
prob_slider=st.sidebar.slider("probability of lottery", 0.01, 2.0, 0.58, 0.01)
max_ratio_slider=st.sidebar.slider("max ratio", 0.01, 5.0, 3.0, 0.01)


#ratio_slider=st.sidebar.slider("ratio", 0.01, 0.1, 0.1, 0.01)

# log toggle button true false
log_toggle = st.sidebar.checkbox("log", value=True)

bayes_lambda = sigma_slider**2 / (nu_slider**2 + sigma_slider**2)

ratio = np.linspace(0, max_ratio_slider, num=1000)

def khaw_cdf(nu, sigma, prob, ratio):
    bayes_lambda = sigma**2 / (nu**2 + sigma**2)
    if log_toggle:
        cdf = norm.cdf(np.log(ratio), loc=bayes_lambda**-1 * np.log(1/prob), scale=np.sqrt(2) * nu)
    else:
        cdf = norm.cdf(ratio, loc=bayes_lambda**-1 * (1/prob), scale=np.sqrt(2) * nu)
    return cdf

def khaw_pdf(nu, sigma, prob, ratio):
    bayes_lambda = sigma**2 / (nu**2 + sigma**2)
    if log_toggle:
        pdf = norm.pdf(np.log(ratio), loc=bayes_lambda**-1 * np.log(1/prob), scale=np.sqrt(2) * nu)
    else:
        pdf = norm.pdf(ratio, loc=bayes_lambda**-1 * (1/prob), scale=np.sqrt(2) * nu)
    return pdf

# make streamlit plot
st.title("Khaw model")

# make mardown area with math

if log_toggle:
    st.markdown(r"""
    $$
    \begin{aligned}
    \mathbb{P}(C = 1) &= \Phi\left(\frac{\log\left(ratio\right) - \lambda^{-1} \log\left(\frac{1}{p}\right)}{\sqrt{2} \nu}\right) \\
    \end{aligned}
    \\
    \text{where }\lambda = \frac{\sigma^2}{\nu^2 + \sigma^2}
    $$
    """)
else:
    st.markdown(r"""
    $$
    \begin{aligned}
    \mathbb{P}(C = 1) &= \Phi\left(\frac{ratio - \lambda^{-1} \frac{1}{p}}{\sqrt{2} \nu}\right) \\
    \end{aligned}
    \\
    \text{where }\lambda = \frac{\sigma^2}{\nu^2 + \sigma^2}
    $$
    """)
bayes_lambda = sigma_slider**2 / (nu_slider**2 + sigma_slider**2)
st.write("inverse of probability = ", round(1/prob_slider,4))
st.write("Threshold, i.e., inverse of probability to the power of 1/lambda= ", round((1/prob_slider)**(bayes_lambda**-1),4))
# make header
st.header("CDF")
# make matplotlib plot
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(ratio, khaw_cdf(nu_slider, sigma_slider, prob_slider, ratio), label="pdf")
if log_toggle:
    plt.axvline(x=(1/prob_slider)**(bayes_lambda**-1), color="red")
else:
    plt.axvline(x=(1/prob_slider), color="red", label="beta")
#html = mpld3.fig_to_html(fig)
#components.html(html, width=800, height=600)
st.pyplot(fig)

st.header("PDF")
fig = plt.figure()
plt.plot(ratio, khaw_pdf(nu_slider, sigma_slider, prob_slider, ratio), label="pdf")
#html = mpld3.fig_to_html(fig)
#components.html(html, width=800, height=600)
st.pyplot(fig)

# add vertical line at beta(1-beta)
#fig, ax = plt.subplots()

#ax.plot(ratio, khaw_pdf(nu_slider, sigma_slider, beta_slider, ratio), label="pdf")
#st.pyplot(fig)




# add vertical line at beta/(1-beta) to plot
#st.line_chart(pd.DataFrame({"ratio": ratio, "beta/(1-beta)": [beta_slider/(1-beta_slider)] * len(ratio)}))


