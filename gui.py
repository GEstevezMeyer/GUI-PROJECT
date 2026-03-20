import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*TensorFlow GPU support is not available on native Windows.*"
)

from textual.app import App
from textual.widgets import Header, Footer, Input
from training import main 
from plots import PlotterTF

class Terminal(App):
    def compose(self):
        yield Header()
        self.input_field = Input(placeholder="Ticket")
        yield self.input_field
        self.plot_widget = PlotterTF()
        yield self.plot_widget
        yield Footer()
    
    async def on_input_submitted(self, event: Input.Submitted):
        self.input_field.disabled = True 
        result,mean,std = main(event.value)
        
        self.plot_widget.history_plot_text(result)
        




if __name__ == "__main__": 
    Terminal().run()