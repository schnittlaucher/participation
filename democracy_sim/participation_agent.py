from typing import TYPE_CHECKING, cast

import numpy as np
from mesa import Agent, Model
from numpy import random
if TYPE_CHECKING:  # Type hint for IDEs
    from democracy_sim.participation_model import ParticipationModel


def combine_and_normalize(arr_1, arr_2, factor):
    # Ensure f is between 0 and 1 TODO: remove this on simulations to speed up
    if not (0 <= factor <= 1):
        raise ValueError("Factor f must be between 0 and 1")

    # Linear combination
    res = factor * arr_1 + (1 - factor) * arr_2
    print(f"un-normalized result: {res}")  # TODO rm
    # Normalize the result
    res_min = res.min()
    return (res - res_min) / (res.max() - res_min + 1e-8)


class VoteAgent(Agent):
    """An agent that has limited knowledge and resources and
    can decide to use them to participate in elections."""

    def __init__(self, unique_id, model, pos, personality, assets=1):
        """ Create a new agent.
        :param unique_id: The unique identifier of the agent.
        :param model: The simulation model of which the agent is part of.
        :type model: ParticipationModel
        :param pos: The position of the agent in the models' grid.
        :type pos: tuple
        :param personality: Represents the agent's preferences among colors.
        :type personality: np.ndarray
        :param assets: The wealth/assets/motivation of the agent.
        """
        # Pass the parameters to the parent class.
        super().__init__(unique_id=unique_id, model=model)
        try:
            row, col = pos
        except ValueError:
            raise ValueError("Position must be a tuple of two integers.")
        self._row = row
        self._col = col
        self._assets = assets
        self.personality = personality
        self.known_cells = []  # ColorCell objects the agent knows (knowledge)
        # Add the agent to the models' agent list
        model.voting_agents.append(self)
        # Place the agent on the grid
        model.grid.place_agent(self, pos)

    @property
    def col(self):
        """Return the col location of this cell."""
        return self._col

    @property
    def row(self):
        """Return the row location of this cell."""
        return self._row

    @property
    def assets(self):
        """Return the assets of this agent."""
        return self._assets

    @assets.setter
    def assets(self, value):
        self._assets = value

    @assets.deleter
    def assets(self):
        del self._assets

    def ask_for_participation(self, area):
        """
        The agent decides
        whether to participate in the upcoming election of a given area.
        :param area: The area in which the election takes place.
        :return: True if the agent decides to participate, False otherwise
        """
        print("Agent", self.unique_id, "decides whether to participate",
              "in election of area", area.unique_id)
        # TODO Implement this (is to be decided upon a learned decision tree)
        return random.choice([True, False])

    def decide_altruism_factor(self, area):
        """
        Uses a trained decision tree to decide on the altruism factor.
        """
        # TODO Implement this (is to be decided upon a learned decision tree)
        # This part is important - also for monitoring - save/plot a_factors
        a_factor = random.uniform(0.0, 1.0)
        print(f"{area}:", "Agent", self.unique_id, "altruism factor:", a_factor)
        return a_factor

    def compute_assumed_opt_dist(self, area):
        """
        Computes a color distribution that the agent assumes to be an optimal
        choice in any election (regardless of whether it exists as a real option
        to vote for or not). It takes "altruistic" concepts into consideration.
        """
        # Compute the "altruism_factor" via a decision tree
        a_factor = self.decide_altruism_factor(area)
        # compute the preference ranking vector as a mix between the agent's
        # own preferences/personality traits and the estimated real distribution
        est_dist = self.estimate_real_distribution(area)
        ass_opt = combine_and_normalize(self.personality, est_dist, a_factor)
        return ass_opt

    def vote(self, area):
        """
        The agent votes in the election of a given area,
        i.e., she returns a preference ranking vector over all options.
        The available options are set in the model.
        :param area: The area in which the election takes place.
        """
        # TODO Implement this (is to be decided upon a learned decision tree)
        # Compute the color distribution that is assumed to be the best choice.
        est_best_dist = self.compute_assumed_opt_dist(area)
        # make sure that r is normalized!
        # (r.min()=0.0 and r.max()=1.0 and all vals x are within [0.0, 1.0]!)
        ##############
        if TYPE_CHECKING:  # Type hint for IDEs
            self.model = cast(ParticipationModel, self.model)
        r = self.model.options[random.choice(self.model.options.shape[0])]
        print("Agent", self.unique_id, "voted:", r)
        return r

    # def step(self):
    #     # Verify agent has some wealth
    #     if self.wealth > 0:
    #         other_a = self.random.choice(self.model.agent_scheduler.agents)
    #         if other_agent is not None:
    #             other_a.wealth += 1
    #             self.wealth -= 1

    # def move(self):
    #     if TYPE_CHECKING: # Type hint for IDEs
    #         self.model = cast(ParticipationModel, self.model)
    #     possible_steps = self.model.grid.get_neighborhood(
    #         self.pos,
    #         moore=True, # Moore vs. von neumann
    #         include_center=False)
    #     new_position = self.random.choice(possible_steps)
    #     self.model.grid.move_agent(self, new_position)
    def estimate_real_distribution(self, area):
        """
        The agent estimates the real color distribution in the area based on
        her own knowledge (self.known_fields).
        """
        relevant_cells = area.filter_cells(self.known_cells)
        known_colors = np.array([cell.color for cell in relevant_cells])
        unique, counts = np.unique(known_colors, return_counts=True)
        distribution = np.zeros(self.model.num_colors)
        distribution[unique] = counts / known_colors.size
        return distribution


class ColorCell(Agent):
    """
    Represents a cell's color
    """

    def __init__(self, unique_id, model, pos, initial_color: int):
        """
        Create a cell, in the given state, at the given row, col position.
        """
        super().__init__(unique_id, model)
        self._row = pos[0]
        self._col = pos[1]
        self._color = initial_color  # The cell's current color (int)
        self._next_color = None
        self._num_agents_in_cell = 0
        self.areas = []
        self.is_border_cell = False

    @property
    def col(self):
        """The col location of this cell."""
        return self._col

    @property
    def row(self):
        """The row location of this cell."""
        return self._row

    @property
    def color(self):
        """The current color of this cell."""
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

    @property
    def num_agents_in_cell(self):
        """The number of agents in this cell."""
        return self._num_agents_in_cell

    @num_agents_in_cell.setter
    def num_agents_in_cell(self, value):
        self._num_agents_in_cell = value

    @num_agents_in_cell.deleter
    def num_agents_in_cell(self):
        del self._num_agents_in_cell

    def add_area(self, area):
        self.areas.append(area)

    def color_step(self):
        """
        Determines the cells' color for the next step
        """
        # _neighbor_iter = self.model.grid.iter_neighbors(
        #     (self._row, self._col), True)
        # neighbors_opinion = Counter(n.get_state() for n in _neighbor_iter)
        # # Following is a tuple (attribute, occurrences)
        # polled_opinions = neighbors_opinion.most_common()
        # tied_opinions = []
        # for neighbor in polled_opinions:
        #     if neighbor[1] == polled_opinions[0][1]:
        #         tied_opinions.append(neighbor)
        #
        # self._next_color = self.random.choice(tied_opinions)[0]
        pass

    def advance(self):
        """
        Set the state of the agent to the next state
        """
        # self._color = self._next_color
        pass
