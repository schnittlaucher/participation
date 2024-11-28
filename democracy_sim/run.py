from mesa.visualization.ModularVisualization import ModularServer
from democracy_sim.participation_model import ParticipationModel
from democracy_sim.model_setup import (model_params as params, canvas_element,
                                       voter_turnout, wealth_chart,
                                       color_distribution_chart)
from democracy_sim.visualisation_elements import *


class CustomModularServer(ModularServer):
    """ This is to prevent double initialization of the model.
    For some reason, the Server resets the model once on initialization
    and again on server launch. """
    def __init__(self, model_cls, visualization_elements,
                 name="Mesa Model", model_params=None, port=None):
        self.initialized = False
        super().__init__(model_cls, visualization_elements, name, model_params,
                         port)

    def reset_model(self):
        if not self.initialized:
            self.initialized = True
            return  # This ensures that the first reset-call is ignored
        super().reset_model()


color_distribution_element = ColorDistributionElement()
personality_distribution = PersonalityDistribution()
election_results = ElectionResultsElement()
steps_text = StepsTextElement()
vto_areas = VoterTurnoutElement()

server = CustomModularServer(
    model_cls=ParticipationModel,
    visualization_elements=[canvas_element, color_distribution_chart,
                            personality_distribution,
                            wealth_chart, voter_turnout, vto_areas,
                            color_distribution_element, election_results,
                            steps_text],
    name="DemocracySim",
    model_params=params,
)

if __name__ == "__main__":
    server.launch(open_browser=True)
