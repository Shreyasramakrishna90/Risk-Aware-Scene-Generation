#!/bin/bash
trap "exit" INT TERM ERR
trap "kill 0" EXIT
k=1
l=1

if [[ -n "$1" ]]; then
    end=$1
else
    end=1
fi

export CARLA_ROOT=/home/scope/Carla/CARLA_0.9.9
export PORT=3000
export ROUTES=leaderboard/data/routes/route_19.xml
export TEAM_AGENT=auto_pilot.py
export TEAM_CONFIG=leaderboard/sample_data
export HAS_DISPLAY=1
total_scenes=$end

for (( j=0; j<=$end-1; j++ ))
  do
    export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
    export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg           # 0.9.9
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
    k=$j
    #total_scenes=$runs

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
    #echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."

done
