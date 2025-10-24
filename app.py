import streamlit as st
import pandas as pd
import functionality as funcs
import visuals
import plotly.graph_objs as go


# setting title of program and tabs
st.title ('Statistical Presentation')

# creating tabs for the program and assigning them to variables tab1 through to tab6
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs (
    ["Import Data", "Auto Analysis", 'Linear Regression', 'Polynomial Regression', 'Logistic Regression',
     'Decision Tree'])


with tab1:
    # allowing user to import a csv file
    imported_data = st.file_uploader ("Choose a CSV file")
    if imported_data is None:

        # if user has not uploaded a dataset then user is given links to example datasets
        with st.expander ('Some Data for Linear Regression'):
            lin_link1 = '[Basic Linear Regression Dataset](https://www.kaggle.com/datasets/tanuprabhu/linear-regression-dataset)'
            lin_link2 = '[Salaries Dataset](https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression)'
            lin_link3 = '[Students Scores Dataset](https://www.kaggle.com/datasets/shubham47/students-score-dataset-linear-regression)'

            st.markdown (lin_link1, unsafe_allow_html=True)
            st.markdown (lin_link2, unsafe_allow_html=True)
            st.markdown (lin_link3, unsafe_allow_html=True)

        with st.expander ('Some Data for Polynomial Regression'):
            pol_link1 = '[Basic Polynomial Regression Dataset](https://www.kaggle.com/datasets/muhammadumarasif/sshapepolynomial-regression)'
            pol_link2 = '[Heights against Weights Dataset](https://www.kaggle.com/datasets/sakshamjn/heightvsweight-for-linear-polynomial-regression)'

            st.markdown (pol_link1, unsafe_allow_html=True)
            st.markdown (pol_link2, unsafe_allow_html=True)

        with st.expander ('Some Data for Logistic Regression'):
            log_link1 = '[Basic Logistic Regression Dataset](https://www.kaggle.com/datasets/dragonheir/logistic-regression)'
            log_link2 = '[Heart Disease Dataset](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression)'
            log_link3 = '[Diabetes Dataset](https://www.kaggle.com/datasets/vikasukani/diabetes-data-set)'

            st.markdown (log_link1, unsafe_allow_html=True)
            st.markdown (log_link2, unsafe_allow_html=True)
            st.markdown (log_link3, unsafe_allow_html=True)

        with st.expander ('Some Data for Decision Tree'):
            tree_link1 = '[Drugs dataset](https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees)'
            tree_link2 = '[Tic-Tac-Toe Dataset](https://www.kaggle.com/datasets/aungpyaeap/tictactoe-endgame-dataset-uci)'
            tree_link3 = '[Term Deposit Dataset](https://www.kaggle.com/datasets/aslanahmedov/predict-term-deposit)'

            st.markdown (tree_link1, unsafe_allow_html=True)
            st.markdown (tree_link2, unsafe_allow_html=True)
            st.markdown (tree_link3, unsafe_allow_html=True)

    if imported_data is not None:

        # when data is uploaded, it is opened and added to the environment
        with open (imported_data.name, "wb") as f:
            f.write (imported_data.getbuffer ())

        # csv file read using pandas and assigned to the variable data and then displayed as a data table in the program
        data = pd.read_csv (imported_data.name)
        st.dataframe (data)

        # allowing user to select which columns to drop
        with st.expander ('You Can Drop Columns That Are Not Relevant'):
            with st.form ('drop columns'):
                dropping = st.multiselect ('What columns to drop', list (data.columns.values))
                drop = st.form_submit_button ('Drop', help='You May Need To Click Twice')

                # if user clicks drop button then the selected columns are dropped from the dataframe and the
                # modified dataframe is saved to the original csv file
                if drop:
                    data = data.drop ([i for i in dropping], axis=1)
                    data.to_csv (imported_data.name, index=False)
        fit1, fit2 = st.columns (2)

        # allow the user to select the purpose of the dataset they just imported and select the target feature if
        # they are doing some sort of classification
        with fit1:
            purpose = st.radio ('Use of Data',
                                ['Analysis', 'Linear Regression', 'Polynomial Regression', 'Logistic Regression',
                                 'Decision Tree'])
        with fit2:
            target = st.selectbox ('Select Target Feature', list (data.columns.values), key='15')

        # making a sidebar which contains different controls depending on what the purpose of the data is
        with st.sidebar:

            if purpose == 'Linear Regression':
                st.subheader ('Hyper parameters for Linear Regression')
                lin_lr = st.text_input ('Learning Rate', key='geu', value=0.001)
                lin_epoch = st.text_input ('Epoch', key='highbrow', value=500)

            elif purpose == 'Polynomial Regression':
                st.subheader ('Hyper parameters for Polynomial Regression')
                degree = st.text_input ('Degree of Polynomial', value=3)

            elif purpose == 'Logistic Regression':
                st.subheader ('Hyper parameters for Logistic Regression')
                log_lr = st.text_input ('Learning Rate', key='mugwump', value=0.001)
                log_epoch = st.text_input ('Epoch', key='housewife', value=500)

            elif purpose == 'Decision Tree':
                st.subheader ('Hyper parameters for Decision Tree')
                min_split = st.text_input ('Minimum Sample Split', value=3)
                max_depth = st.text_input ('Maximum Depth of Tree', value=5)


with tab2:

    if imported_data is not None and purpose == 'Analysis':
        st.dataframe (data)
        st.write ('Plot Charts And Get Insight Into The Data With Natural Language Commands')
        attributes = data.columns.values # list of the column names of the dataset


        def get_feature(arr):
            return attributes[arr[0]], attributes[arr[1]]

        twill, prune = st.columns (2)

        with twill:
            text = st.text_input ('Write Command')

        with prune:
            st.write ('')
            st.write ('')
            nlp = st.button ('Run Command', help='try to make sure everything spelt right before running')

        if nlp:
            # when button is clicked command is processed
            function, features = funcs.fake_nlp (text, attributes)
            features = get_feature (features)
            result = funcs.commands[function] (data, features)
            # if the result of the command is a chart is displays a chart, else it displays the result as a metric
            if type (result) == go._figure.Figure:
                st.plotly_chart (result)

            else:
                st.metric(result[0], result[1])

with tab3:
    if imported_data is not None and purpose == 'Linear Regression':
        st.dataframe (data.T) # displaying dataframe
        run_linear = st.button ('Run')

        if run_linear:
            # Calculating the linear regression model with the given parameters
            lin_reg = funcs.calc_lin_reg (int (lin_epoch), float (lin_lr), data)
            # extracting the linear regression formula as a string and displaying as markdown
            string_lin = lin_reg[0]
            st.markdown (string_lin)
            # plotting the linear regression line
            st.plotly_chart (visuals.regline (data, string_lin))
            lin_reg_accuracy = round (lin_reg[1], 3)
            st.metric ('The RMSE:', lin_reg_accuracy)

with tab4:
    if imported_data is not None and purpose == 'Polynomial Regression':
        st.dataframe (data.T) # displaying dataframe
        run_poly = st.button ('Run', key='segue')

        if run_poly:
            # Calculating the polynomial regression model with the given parameters
            poly_reg = funcs.calc_poly_reg (int (degree), data)
            # extracting the polynomial regression formula as a string and displaying as markdown
            string_poly = poly_reg[0]
            st.markdown (string_poly)
            # plotting the polynomial regression line
            st.plotly_chart (visuals.regline (data, string_poly))
            poly_reg_accuracy = round (poly_reg[1], 3)
            st.metric ('The RMSE:', poly_reg_accuracy)

with tab5:
    if imported_data is not None and purpose == 'Logistic Regression':
        st.write ('The Target Feature')
        # Display the target feature data as DataFrame
        st.dataframe (data[[target]].T)
        run_log = st.button ('Run', key='regular', help='make sure target feature is correct')

        if run_log:
            # Calculate the logistic regression model with the given parameters and target feature
            log_reg = funcs.calc_log_reg (int (log_epoch), float (log_lr), data, target)
            # Extract the predicted values from the logistic regression model and create a DataFrame
            log_data = pd.DataFrame (log_reg[0], columns=[target])
            # Turn the 1s and 0s returned by logistic regression back into non-numerical answers and display
            f = lambda x: log_reg[2][x]
            log_data[target] = log_data[target].apply (f)
            st.write ('Predicted Values')
            st.dataframe (log_data.T)
            # Display accuracy and create a confusion matrix of results
            st.metric ('Accuracy', str (round (log_reg[1], 2)) + "%")
            st.plotly_chart (visuals.confusion_matrix (log_reg[0], data[target].values))

with tab6:
    if imported_data is not None and purpose == 'Decision Tree':
        st.write ('The Target Feature')
        # Display the target feature data as DataFrame
        st.dataframe (data[[target]].T)
        run_dec = run_log = st.button ('Run', key='reps', help='make sure target feature is correct')

        if run_dec:
            # building the decision tree with the given parameters and target feature
            dec_tree = funcs.calc_dec_tree (int (min_split), int (max_depth), data, target)
            # Extract the predicted values from the decision tree and create a DataFrame
            dec_data = pd.DataFrame (dec_tree[0], columns=[target])
            # Turn the 1s and 0s returned by decision tree back into non-numerical answers and display
            fx = lambda x: dec_tree[2][x]
            dec_data[target] = dec_data[target].apply (fx)
            st.write ('Predicted Values')
            st.dataframe (dec_data.T)
            # Display accuracy and create a confusion matrix of results
            st.metric ('Accuracy', str (round (dec_tree[1], 2)) + "%")
            st.plotly_chart (visuals.confusion_matrix (dec_tree[0], data[target].values))
