#!/bin/bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT
k=1
l=1
if [[ -n "$1" ]]; then
    end=$1
else
    end=2
fi
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
optimization_algorithm="random" #"bayesian_optimization"
textx generate Scenario-Description-Updated/scene/carla.tx --target dot
textx generate Scenario-Description-Updated/scene/scene-model.carla --grammar Scenario-Description-Updated/scene/carla.tx --target dot
textx generate Scenario-Description-Updated/scene/agent-model.carla --grammar Scenario-Description-Updated/scene/carla.tx --target dot

export CARLA_ROOT=/CARLA_0.9.9
export PORT=2000
export ROUTES=leaderboard/data/routes/route_19.xml
export TEAM_AGENT=image_agent.py
export TEAM_CONFIG=carla_project/model.ckpt
export HAS_DISPLAY=1

total_scenes=$end
#for j in {0..$runs}
for (( j=0; j<=$end-1; j++ ))
  do
    export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
    #export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg           # 0.9.8
    export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg           # 0.9.8
    export PYTHONPATH=$PYTHONPATH:leaderboard
    export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
    export PYTHONPATH=$PYTHONPATH:scenario_runner
    export PYTHONPATH=$PYTHONPATH:carla_project

    if [ -d "$TEAM_CONFIG" ]; then
        CHECKPOINT_ENDPOINT="$TEAM_CONFIG/$(basename $ROUTES .xml).txt"
    else
        CHECKPOINT_ENDPOINT="$(dirname $TEAM_CONFIG)/$(basename $ROUTES .xml).txt"
    fi
    i=$j
    python3 Scenario-Description-Updated/scene/scene_interpreter.py \
    --simulation_num=${i}\
    --scene_num=${l}\
    --optimizer=${optimization_algorithm}\
    --total_scenes=${total_scenes}

    python3 leaderboard/leaderboard/leaderboard_evaluator.py \
    --challenge-mode \
    --track=dev_track_3 \
    --scenarios=leaderboard/data/all_towns_traffic_scenarios_public.json  \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --routes=${ROUTES} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --port=${PORT} \
    --record='/home/carla/data'\
    --simulation_number=${j}\
    --scene_number=${k}
    echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."
    #k=$((k+=1))
    #docker cp carla_simulator:/home/carla/data/collision_data.log /home/scope/Carla/carla-dockers/simulation-data
done
python3 leaderboard/team_code/plot-stats.py
