import Dictionarys


def get_director_coeficient(director_name):
    if director_name in Dictionarys.directors:
        position = Dictionarys.directors[director_name]
        coeficient = 0.3 + (75 - position) * 0.7 / 75

    else:
        coeficient = 0
    return coeficient

def get_actor_coeficient(actor_name):
    if actor_name in Dictionarys.actors:
        position = Dictionarys.actors[actor_name]
        coeficient = 0.3 + (300 - position) * 0.7 / 300

    else:
        coeficient = 0
    return coeficient

