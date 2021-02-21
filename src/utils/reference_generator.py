from typing import List


def generate_reference(initial_temp: float,
                       heatup_times: List[int],
                       anneal_times: List[int],
                       target_temps: List[float]):
    assert len(heatup_times) == len(anneal_times)
    assert len(heatup_times) == len(target_temps)

    # compute entire time stamps

    entire_time = sum(heatup_times) + sum(anneal_times)

    ref_temps = []
    target_temps = [initial_temp] + target_temps
    for i in range(len(target_temps) - 1):
        start_temp = target_temps[i]
        end_temp = target_temps[i + 1]
        heatup_time = heatup_times[i]

        slope = (end_temp - start_temp) / heatup_time

        anneal_time = anneal_times[i]
        temps = [start_temp + slope * i for i in range(heatup_time)] + [end_temp] * anneal_time
        ref_temps.extend(temps)

    assert entire_time == len(ref_temps)
    return ref_temps
