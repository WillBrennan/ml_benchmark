from typing import Tuple
import pathlib
import logging
import subprocess

import torch
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
from torchvision.models import mnasnet1_3, mnasnet1_0
from torchvision.models import shufflenet_v2_x1_5, regnet_y_400mf, efficientnet_b0
import pandas as pd
import fire
import seaborn as sns

from .models import BiSeNetV2, FastSCNN, FASSDNet

Resolution = Tuple[int, int]


class BenchmarkML:
    def generate_onnx(self):
        fn_models = [
            resnet18,
            resnet34,
            resnet50,
            mobilenet_v3_large,
            mobilenet_v3_small,
            mnasnet1_0,
            mnasnet1_3,
            BiSeNetV2,
            FastSCNN,
            FASSDNet,
            shufflenet_v2_x1_5,
            regnet_y_400mf,
            efficientnet_b0,
        ]

        for fn_model in fn_models:
            model_name = str(fn_model.__name__)
            onnx_path = f"{model_name}.onnx"

            logging.info("saving %s to %s", model_name, onnx_path)

            with torch.no_grad():
                model = fn_model(weights=None)
                dummy_input = torch.zeros(1, 3, 480, 640)

                dynamic_axes = {
                    "input": {
                        0: "batch_size",
                        2: "image_height",
                        3: "image_width",
                    }
                }

                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes=dynamic_axes,
                )

    def benchmark_onnx(
        self,
        base_dir: pathlib.Path = pathlib.Path.cwd(),
        resolutions: list[Resolution] = [
            (640, 480),
            (1024, 768),
            (1920, 1056),
            (3840, 2144),
        ],
    ):
        onnx_paths = list(base_dir.rglob("*.onnx"))
        logging.info("found %i onnx files in %s", len(onnx_paths), base_dir)

        all_data_path = base_dir / "benchmarking_data.csv"
        if all_data_path.exists():
            all_data = pd.read_csv(all_data_path)
        else:
            all_data = pd.DataFrame()

        for onnx_path in onnx_paths:
            logging.info("benchmarking %s", onnx_path.stem)

            for precision in "", "--fp16", "--fp16 --int8":
                precision_file = precision.replace(" ", "_").replace("-", "")
                logging.info("benchmarking precision %s", precision)

                engine_path = base_dir / f"{onnx_path.stem}_{precision_file}.trt.engine"

                if not engine_path.exists():
                    logging.info("saving engine to %s", engine_path)

                    cmd = f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_path} {precision}"
                    cmd += " --minShapes=input:1x3x480x640 --maxShapes=input:1x3x2144x3840"
                    cmd += f" --skipInference --saveEngine={engine_path}"

                    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)

                for width, height in resolutions:
                    logging.info("benchmarking with resolution %s x %s", width, height)
                    export_path = base_dir / f"benchmark_{onnx_path.stem}_{precision_file}_h{height}_w{width}.json"

                    if not export_path.exists():
                        cmd = f"/usr/src/tensorrt/bin/trtexec --shapes=input:1x3x{height}x{width}"
                        cmd += " --useSpinWait --avgRuns=50 --noDataTransfers"
                        cmd += f" --loadEngine={engine_path} --exportTimes={export_path}"
                        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)

                        logging.info("saving results")

                    data = pd.read_json(export_path)
                    data["width"] = width
                    data["height"] = height
                    data["model"] = onnx_path.stem
                    data["fp16"] = "fp16" in precision
                    data["int8"] = "int8" in precision

                    all_data = pd.concat([data, all_data])
                    all_data.to_csv(all_data_path)

    def graph(self, path="benchmarking_data.csv"):
        model_group = ["model", "width", "height", "fp16", "int8"]
        columns = model_group + ["computeMs"]

        data = pd.read_csv(path)
        data = data[columns]

        stats = data.groupby(model_group).mean().reset_index()
        stats.to_csv("summary.csv")

        stats["resolution"] = stats["width"].astype(str) + "x" + stats["height"].astype(str)
        stats["fps"] = 1000 / stats["computeMs"]

        stats_fp16 = stats[stats["fp16"]]
        stats_int8 = stats[stats["fp16"] & stats["int8"]]

        sns.set_theme(rc={"figure.figsize": (11.7, 8.27)})

        ax = sns.lineplot(data=stats_fp16, x="resolution", y="fps", hue="model", errorbar=None, palette="bright")
        ax.figure.savefig("stats_fp16.png")
        ax.figure.clear()

        ax = sns.lineplot(data=stats_int8, x="resolution", y="fps", hue="model", errorbar=None, palette="bright")
        ax.figure.savefig("stats_int8.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(BenchmarkML)
