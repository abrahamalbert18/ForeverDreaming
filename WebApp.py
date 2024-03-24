# Importing modules
from dash import Dash, html, dcc, callback, Input, Output, State
import GenerateScripts as gs

# App instantiation
app = Dash(name="Forever Dreaming")

# App layout
app.layout = html.Div(
        [
                html.H1("Welcome to Forever Dreaming!"),
                html.H2("Start a prompt here and our model will finish your "
                        "text."),
                dcc.Textarea(id="TextArea", value="Ross:", maxLength=50,
                             style={"width" : "50%", "height" : 100}),
                html.Button('Generate',
                            id='GenerateButton', n_clicks=0),
                html.H2("Model Output:\n"),
                dcc.Textarea(id="OutputTextArea", value="Ross:",
                             style={"width": "50%", "height": 300}),
        ]
)

# Callbacks
@callback(
        Output(component_id="OutputTextArea", component_property="value"),
        Input(component_id="GenerateButton", component_property="n_clicks"),
        State(component_id="TextArea", component_property="value"),
)
def updateTextArea(n_clicks, value):
    if n_clicks > 0:
        # print("value", value)
        generatedText = f"{value}"
        source, target = gs.tokenizeSourceAndTarget(sentence=value)
        generatedText = gs.generateText(source, target)
        return generatedText

if __name__ == '__main__':
    app.run_server(debug=True)