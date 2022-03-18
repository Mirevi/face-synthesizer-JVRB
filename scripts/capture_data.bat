cd ../
"./.venv/Scripts/python.exe" ./capture_data.py --image_amount 50 --image_id_offset 0 --visualize --name new_capture
"./.venv/Scripts/python.exe" ./capture_data.py --image_amount 1750 --image_id_offset 0 --name new_capture --overwrite
cd scripts