from glob import glob

from setuptools import setup


package_name = "mycobot_description"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml", "LICENSE"],
        ),
        (
            "share/" + package_name + "/urdf/mycobot_280_jn",
            glob("urdf/mycobot_280_jn/*"),
        ),
        (
            "share/" + package_name + "/urdf/adaptive_gripper",
            glob("urdf/adaptive_gripper/*"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Raj Indulkar",
    maintainer_email="rajindulkar7@gmail.com",
    description=(
        "myCobot 280 Jetson Nano adaptive-gripper robot description"
    ),
    license="BSD-2-Clause",
)
