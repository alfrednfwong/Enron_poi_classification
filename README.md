# Enron_poi_classification

Final project for Intro to Machine Learning at Udacity
by Alfred Wong
In this study we aim to identify "persons of interest" in the Enron scandal. Enron was an energy company in the US, which after its irregular accounting practices came to light, filed for bankruptcy in 2001.

Many people on the board and in the management were investigated, and some of them either: were indicted; settled without admitting guilt; or testified in exchange for immunity. And these are the individuals we are interested in.

The dataset we use to help identify these persons of interest(POIs) consists of 146 observations, each being a person (with one exception) and 21 features. The features are either financial information of the persons from findlaw.com, or email counts, which come from a publically available dataset that includes more than half a million emails between people involved with Enron.

The main feature here is poi, a boolean label that indicates whether the person is a POI. Among the 145 persons in the data set, 18 are identified as POIs.