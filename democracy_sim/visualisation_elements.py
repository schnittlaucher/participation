import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, cast
from mesa.visualization import TextElement
import matplotlib.patches as patches
from model_setup import _COLORS
import base64
import math
import io

_COLORS[0] = "LightGray"

def save_plot_to_base64(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f'<img src="data:image/png;base64,{image_base64}" />'


class AreaStats(TextElement):
    def render(self, model):
        # Only render if show_area_stats is enabled
        step = model.scheduler.steps
        if not model.show_area_stats or step == 0:
            return ""

        # Fetch data from the datacollector
        data = model.datacollector.get_agent_vars_dataframe()
        color_distribution = data['ColorDistribution'].dropna()
        dist_to_reality = data['DistToReality'].dropna()
        election_results = data['ElectionResults'].dropna()

        # Extract unique area IDs (excluding the global area)
        area_ids = color_distribution.index.get_level_values(1).unique()[1:]
        num_colors = len(color_distribution.iloc[0])
        num_areas = len(area_ids)

        # Create subplots with two columns (two plots per area).
        fig, axes = plt.subplots(nrows=num_areas, ncols=2,
                                 figsize=(8, 4 * num_areas), sharex=True)

        for area_id in area_ids:
            row = area_id
            # Left plot: color distribution and distance to reality value
            ax1 = axes[row, 0]
            area_data = color_distribution.xs(area_id, level=1)
            for color_idx in range(num_colors):
                color_data = area_data.apply(lambda x: x[color_idx])
                ax1.plot(color_data.index, color_data.values,
                         color=_COLORS[color_idx])
            a_data = dist_to_reality.xs(area_id, level=1)
            ax1.plot(a_data.index, a_data.values, color='Black', linestyle='--')
            ax1.set_title(f'Area {area_id} \n'
                          f'--- deviation from voted distribution')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Color Distribution')

            # Right plot: election results
            ax2 = axes[row, 1]
            area_data = election_results.xs(area_id, level=1)
            for color_id in range(num_colors):
                color_data = area_data.apply(lambda x: list(x).index(
                    color_id) if color_id in x else None)
                ax2.plot(color_data.index, color_data.values,
                         label=f'Color {color_id}', color=_COLORS[color_id])
            ax2.set_title(f'Area {area_id} \n')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Election Results (rank values)')
            ax2.invert_yaxis()

        plt.tight_layout()
        combined_plot = save_plot_to_base64(fig)

        return combined_plot


class PersonalityDistribution(TextElement):

    def __init__(self):
        super().__init__()
        self.personality_distribution = None
        self.pers_dist_plot = None

    def create_once(self, model):
        if TYPE_CHECKING:
            model = cast('ParticipationModel', model)
        # Fetch data
        dists = model.personality_distribution
        personalities = model.personalities
        num_personalities = personalities.shape[0]
        num_agents = model.num_agents
        colors = _COLORS[:model.num_colors]


        fig, ax = plt.subplots(figsize=(6, 4))
        heights = dists * num_agents
        bars = ax.bar(range(num_personalities), heights, color='skyblue')

        for bar, personality in zip(bars, personalities):
            height = bar.get_height()
            width = bar.get_width()

            for i, color_idx in enumerate(personality):
                coords = (bar.get_x() + width / 2 - 0.4 + i * 0.2, height)
                rect = patches.Rectangle(coords, 0.2, 5,
                                         color=colors[color_idx])
                ax.add_patch(rect)

        ax.set_xlabel('"Personality" ID')
        ax.set_ylabel('Number of Agents')
        ax.set_title('Distribution of Personalities among Agents')

        plt.tight_layout()
        self.pers_dist_plot = save_plot_to_base64(fig)

    def render(self, model):
        # Only create a new plot at the start of a simulation
        if model.scheduler.steps == 0:
            self.create_once(model)
        return self.pers_dist_plot


class VoterTurnoutElement(TextElement):
    def render(self, model):
        # Only render if show_area_stats is enabled
        step = model.scheduler.steps
        if not model.show_area_stats or step == 0:
            return ""
        # Fetch data from the datacollector
        data = model.datacollector.get_agent_vars_dataframe()
        voter_turnout = data['VoterTurnout'].dropna()

        # Extract unique area IDs
        area_ids = voter_turnout.index.get_level_values(1).unique()

        # Create a single plot
        fig, ax = plt.subplots(figsize=(8, 6))

        for area_id in area_ids:
            area_data = voter_turnout.xs(area_id, level=1)
            ax.plot(area_data.index, area_data.values, label=f'Area {area_id}')

        ax.set_title('Voter Turnout by Area Over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel('Voter Turnout (%)')
        ax.legend()

        return save_plot_to_base64(fig)


class MatplotlibElement(TextElement):
    def render(self, model):
        # Only render if show_area_stats is enabled
        step = model.scheduler.steps
        if not model.show_area_stats or step == 0:
            return ""
        # Fetch data from the datacollector
        data = model.datacollector.get_model_vars_dataframe()
        collective_assets = data["Collective assets"]

        # Create a plot
        fig, ax = plt.subplots()
        ax.plot(collective_assets, label="Collective assets")
        ax.set_title("Collective Assets Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Collective Assets")
        ax.legend()

        return save_plot_to_base64(fig)

class StepsTextElement(TextElement):
    def render(self, model):
        step = model.scheduler.steps
        # TODO clean up
        first_agents = [str(a) for a in model.voting_agents[:5]]
        text = (f"Step: {step} | cells: {len(model.color_cells)} | "
                f"areas: {len(model.areas)} | First 5 voters of "
                f"{len(model.voting_agents)}: {first_agents}")
        return text
