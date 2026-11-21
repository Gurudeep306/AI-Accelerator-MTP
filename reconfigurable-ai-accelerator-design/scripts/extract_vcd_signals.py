import re

scope = []
signals = []

with open("phase2_accelerator_core.vcd") as f:
    for line in f:
        line = line.strip()

        if line.startswith("$scope"):
            parts = line.split()
            scope.append(parts[2])

        elif line.startswith("$upscope"):
            if scope:
                scope.pop()

        elif line.startswith("$var"):
            parts = line.split()
            width = parts[2]
            identifier = parts[3]
            name = parts[4]

            signals.append(
                f"{'.'.join(scope)} | {name} | width={width} | id={identifier}"
            )

with open("vcd_signal_list.txt", "w") as f:
    f.write("\n".join(signals))

print(f"Extracted {len(signals)} signals.")
