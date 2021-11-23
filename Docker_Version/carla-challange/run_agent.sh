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

if [[ -n "$2" ]]; then
    exploration_runs=$2
else
    exploration_runs=0
fi

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export CARLA_ROOT=/CARLA_0.9.9
export PORT=3000
export ROUTES=leaderboard/data/routes/route_19.xml
export TEAM_AGENT=image_agent.py
export TEAM_CONFIG=carla_project/model.ckpt
export HAS_DISPLAY=1
export PYTHONPATH=$PYTHONPATH:/CARLA_0.9.9/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/CARLA_0.9.9/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg           # 0.9.9
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export PYTHONPATH=$PYTHONPATH:carla_project
total_scenes=$end

#Initialize carla
$CARLA_PATH/CarlaUE4.sh -quality-level=Epic -world-port=3000 -resx=800 -resy=600 -opengl &
PID=$!
echo "Carla PID=$PID"
sleep 30

textx generate sdl/scene/carla.tx --target dot
textx generate sdl/scene/scene-model.carla --grammar sdl/scene/carla.tx --target dot
textx generate sdl/scene/agent-model.carla --grammar sdl/scene/carla.tx --target dot


for (( j=0; j<=$end-1; j++ ))
  do

    if [ -d "$TEAM_CONFIG" ]; then
        CHECKPOINT_ENDPOINT="$TEAM_CONFIG/$(basename $ROUTES .xml).txt"
    else
        CHECKPOINT_ENDPOINT="$(dirname $TEAM_CONFIG)/$(basename $ROUTES .xml).txt"
    fi
    i=$j
    k=$j
    #total_scenes=$runs
    cutoff=$exploration_runs
    python3.7 sdl/scene/scene_interpreter.py \
    --project_path=''\
    --simulation_num=${i}\
    --scene_num=${l}\
    --optimizer=${optimization_algorithm}\
    --total_scenes=${total_scenes}\
    --exploration=${exploration_runs}\

    python3.7 leaderboard/leaderboard/leaderboard_evaluator.py \
    --challenge-mode \
    --track=dev_track_3 \
    --scenarios=leaderboard/data/all_towns_traffic_scenarios_public.json  \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --routes=${ROUTES} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --port=${PORT} \
    --record=''\
    --simulation_number=${j}\
    --scene_number=${k}\
    --project_path=''

done
# python3 leaderboard/team_code/plot-stats.py \
# --cutoff=${cutoff} \
echo "Done CARLA Simulations"

pkill -f "CarlaUE4"
