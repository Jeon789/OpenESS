#!/bin/python

"""
modbus_client_oper1_kafka+protectionmap.py : 태양광 ESS 데이터 수집을 위한 코드 (카프카 consumer 및 특이치 저장)

        ---------------------------------------------------------------------------
        Copyright(C) 2022, 윤태일 / KETI / taeil777@keti.re.kr

        아파치 라이선스 버전 2.0(라이선스)에 따라 라이선스가 부여됩니다.
        라이선스를 준수하지 않는 한 이 파일을 사용할 수 없습니다.
        다음에서 라이선스 사본을 얻을 수 있습니다.

        http://www.apache.org/licenses/LICENSE-2.0

        관련 법률에서 요구하거나 서면으로 동의하지 않는 한 소프트웨어
        라이선스에 따라 배포되는 것은 '있는 그대로' 배포되며,
        명시적이든 묵시적이든 어떠한 종류의 보증이나 조건도 제공하지 않습니다.
        라이선스에 따른 권한 및 제한 사항을 관리하는 특정 언어는 라이선스를 참조하십시오.
        ---------------------------------------------------------------------------

        ---------------------------------------------------------------------------
        The MIT License

        Copyright(C) 2022, 윤태일 / KETI / taeil777@keti.re.kr

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in
        all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
        THE SOFTWARE.
        ---------------------------------------------------------------------------

최신 테스트 버전 : 1.0.0 ver
최신 안정화 버전 : 1.0.0 ver

시온유 태양광 ESS데이터 수집을 위한 코드입니다.

kafak 프로듀서 클러스터에서 consumer를 통해 데이터를 수집하고 timescaledb 를 통해 데이터를 입력합니다.

protection 맵에 따라서 특이치에 대한 저장은 별도로 이루어집니다.

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.
       

"""


from pyModbusTCP.client import ModbusClient

import time
import timescale_input_test
import datetime

from multiprocessing import Process
import threading
from pytz import timezone
import numpy as np
import logs
import os
import threading

from kafka import KafkaConsumer
from json import loads


if __name__ == "__main__":

    operation_site = "operation1"
    topic_name = "sionyu"
    consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=[
            "카프카클러스터1",
            "카프카클러스터2",
            "카프카클러스터3",
        ],
        auto_offset_reset="latest",
        enable_auto_commit=True,
        # group_id=None,
        # group_id="gcp_oper1",
        group_id="local_oper1",
        value_deserializer=lambda x: loads(x.decode("utf-8")),
        consumer_timeout_ms=1000,
    )
    # log_path = "/home/keti_iisrc/test/log/"
    log_path = "/home/keti/test/log/"
    main_logger = logs.get_logger("operation1", log_path, "operation1.json")

    while True:
        try:
            while True:
                print("[begin] get consumer list")

                for message in consumer:

                    seoultime = datetime.datetime.fromtimestamp(
                        message.timestamp / 1000
                    ).replace(microsecond=0)

                    # seoultime = datetime.now(timezone("Asia/Seoul")).replace(microsecond=0)

                    print(seoultime)

                    Bank_data = message.value["Bank_data"]
                    Rack_data = message.value["Rack_data"]
                    PCS_data = message.value["PCS_data"]
                    ETC_data = message.value["ETC_data"]

                    timescale_feature = timescale_input_test.timescale(
                        ip="IP",
                        # ip="localhost",
                        port="port",
                        username="username",
                        password="password",
                        dbname="dbname",
                    )

                    # bank 45 : MASTER_RACK_COMMUNICATION_FAULT
                    if Bank_data[43] == 1:
                        message = """마스터 Rack BMS 통신에러""".format()
                        timescale_feature.outlier_detection(
                            seoultime=seoultime,
                            error_code=14,
                            error_level=2,
                            bank_id=1,
                            rack_id=0,
                            operating_site=1,
                            description=message,
                        )

                    for i in range(8):

                        if Rack_data[i][26] == 1:
                            message = """랙 온도 불균형 경고, cell 온도편차 : {value}""".format(
                                value=Rack_data[i][16]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=1,
                                error_level=1,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][27] == 1:
                            message = """랙 저온 경고, 최소 cell 온도 : {value1}, 최소 cell 온도 위치 : {value2} """.format(
                                value1=Rack_data[i][14], value2=Rack_data[i][15]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=2,
                                error_level=1,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )

                        if Rack_data[i][28] == 1:
                            message = """랙 고온 경고, 최대 cell 온도 : {value1}, 최대 cell 온도 위치 : {value2} """.format(
                                value1=Rack_data[i][12], value2=Rack_data[i][13]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=3,
                                error_level=1,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][29] == 1:
                            message = """랙 전압 불균형 경고, cell 전압편차 : {value}""".format(
                                value=Rack_data[i][10]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=4,
                                error_level=1,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][30] == 1:
                            message = """랙 저전압 경고, 최소 cell 전압 : {value1}, 최소 cell 전압 위치 : {value2}""".format(
                                value1=Rack_data[i][8], value2=Rack_data[i][9]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=5,
                                error_level=1,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )

                        if Rack_data[i][31] == 1:
                            message = """랙 과전압 경고, 최대 cell 전압 : {value1}, 최대 cell 전압 위치 : {value2}""".format(
                                value1=Rack_data[i][6], value2=Rack_data[i][7]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=6,
                                error_level=1,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][32] == 1:
                            message = """랙 충전 과전류 경고, 랙 전류 : {value}""".format(
                                value1=Rack_data[i][5]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=7,
                                error_level=1,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][33] == 1:
                            message = """랙 방전 과전류 경고, 랙 전류 : {value}""".format(
                                value1=Rack_data[i][5]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=8,
                                error_level=1,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][34] == 1:
                            message = """랙 온도 불균형 장애, cell 온도편차 : {value}""".format(
                                value=Rack_data[i][16]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=1,
                                error_level=2,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][35] == 1:
                            message = """랙 저온 장애, 최소 cell 온도 : {value1}, 최소 cell 온도 위치 : {value2} """.format(
                                value1=Rack_data[i][14], value2=Rack_data[i][15]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=2,
                                error_level=2,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][36] == 1:
                            message = """랙 고온 장애, 최대 cell 온도 : {value1}, 최대 cell 온도 위치 : {value2} """.format(
                                value1=Rack_data[i][12], value2=Rack_data[i][13]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=3,
                                error_level=2,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][37] == 1:
                            message = """랙 전압 불균형 장애, cell 전압편차 : {value}""".format(
                                value=Rack_data[i][10]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=4,
                                error_level=2,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][38] == 1:
                            message = """랙 저전압 장애, 최소 cell 전압 : {value1}, 최소 cell 전압 위치 : {value2}""".format(
                                value1=Rack_data[i][8], value2=Rack_data[i][9]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=5,
                                error_level=2,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )

                        if Rack_data[i][39] == 1:
                            message = """랙 과전압 장애, 최대 cell 전압 : {value1}, 최대 cell 전압 위치 : {value2}""".format(
                                value1=Rack_data[i][6], value2=Rack_data[i][7]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=6,
                                error_level=2,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][40] == 1:
                            message = """랙 충전 과전류 장애, 랙 전류 : {value}""".format(
                                value1=Rack_data[i][5]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=7,
                                error_level=2,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][41] == 1:
                            message = """랙 방전 과전류 장애, 랙 전류 : {value}""".format(
                                value1=Rack_data[i][5]
                            )
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=8,
                                error_level=2,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][42] == 1:
                            message = """랙 충전 컨텍터 고장 """.format()
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=9,
                                error_level=2,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][43] == 1:
                            message = """랙 방전 컨텍터 고장 """.format()
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=10,
                                error_level=2,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )

                        if Rack_data[i][44] == 1:
                            message = """랙 (-) 퓨즈 장애 """.format()
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=11,
                                error_level=2,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )

                        if Rack_data[i][45] == 1:
                            message = """랙 (+) 퓨즈 장애 """.format()
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=12,
                                error_level=2,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )
                        if Rack_data[i][46] == 1:
                            message = """트레이 - 랙 통신 이상 """.format()
                            timescale_feature.outlier_detection(
                                seoultime=seoultime,
                                error_code=13,
                                error_level=2,
                                bank_id=1,
                                rack_id=i + 1,
                                operating_site=1,
                                description=message,
                            )

                    print("Bank_data: ", Bank_data)
                    print("Rack_data: ", Rack_data)
                    print("PCS_data: ", PCS_data)
                    print("ETC_data: ", ETC_data)

                    timescale = timescale_input_test.timescale(
                        ip="ip",
                        port="port",
                        username="username",
                        password="password",
                        dbname="dbname",
                    )
                    timescale.Bank_input_data(seoultime, Bank_data, operation_site)
                    timescale.Rack_input_data(seoultime, Rack_data, operation_site)
                    timescale.PCS_input_data(seoultime, PCS_data, operation_site)
                    timescale.ETC_input_data(seoultime, ETC_data, operation_site)

                    print("[end] get consumer list")

                time.sleep(5)
        except Exception as e:
            log_message = """kafka consumer error : {error}""".format(error=e)
            main_logger.error(log_message)
            time.sleep(2)
            continue
