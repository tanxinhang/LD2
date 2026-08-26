# Run the 25-seed gap-coverage focus-450 confirmation (Amendment 3).
# Usage: powershell -File tools/run_gapcov_25seed.ps1
$ErrorActionPreference = "Stop"
$root = "D:\BYLW\LD3"
$py = "$root\pytrch_ven\Scripts\python.exe"
$seeds = "1019466100,1466409252,272618232,2005825564,1969862817,1914864024," +
         "1432484627,1245781959,530968976,2063700163,597462755,528245837," +
         "1992563642,1080924982,1329066063,330692561,1224507828,529576084," +
         "1609364221,647973436,1834410891,1491304537,1755461577,1704378734,404347358"
& $py "$root\scripts\run_mappo.py" `
    --config config\exp_800_k12q12_distributed_v2_dynamic_u2u_robustbelief_gapcoverage_pilot.yaml `
    --warm-start results\_audit_k10_tail40_safe_robust2\best_restored.pt `
    --seed 725 --episodes 0 `
    --final-eval-seeds $seeds `
    --out-dir results\_conf25_gapcov_focus450