#!/usr/bin/env python
# coding: utf-8

# # 将采集的trs文件转为numpy常用的npz格式


# convert trs to npy
import numpy as np
import matplotlib.pyplot as plt
import os
#get_ipython().run_line_magic('matplotlib', 'inline')

import logging
import numpy as np
from struct import pack, unpack

def progress(count, total, job_name=''):
    """Print the colored progress bar"""

    if count != (total - 1):
        if count % (int(total / 10)) != 0:
            return
    width = 50
    percents = int(round(100.0 * count / float(total), 1))
    end = ''
    status = "Running..."
    if percents == 100:
        end = '\n'
        status = 'Finished.'
    # high contras blue is 96 in 8 basic color system
    bar = '\x1b[{};1m{}\x1b[0m'.format(96, (int(percents/2)) * '#')
    white = (width - int(percents/2)) * ' '
    out_str = '\r[{}{}] \x1b[94;1m{}% \x1b[93;1m{} {}\x1b[0m'.format(bar, white, percents, job_name, status)
    print(out_str, end=end)


class CommonFile(object):
    __filePath = ''
    __byteNum = 0
    __fileHandler = None

    def __init__(self, filePath):
        self.__filePath = filePath

    @property
    def byteNum(self):
        return self.__byteNum

    def openFile(self, mode):
        self.__fileHandler = open(self.__filePath, mode)

    def readbyte(self, num):
        byte_re = self.__fileHandler.read(num)
        self.__byteNum += num
        return byte_re

    def readint(self, num=4):
        byte_re = self.__fileHandler.read(num)
        self.__byteNum += num
        return int.from_bytes(byte_re, 'little')

    def readstr(self, num):
        byte_re = self.__fileHandler.read(num)
        self.__byteNum += num
        return byte_re.decode()

    def readfloat(self, num=4):
        byte_re = self.__fileHandler.read(num)
        self.__byteNum += num
        return float.fromhex(byte_re.hex())

    def seekfile(self, num=0):
        self.__fileHandler.seek(num, 0)
        return True

    def writeByte(self, value):
        self.__fileHandler.write(value)

    def closeFile(self):
        self.__fileHandler.close()


class TrsHandler(object):
    """.trs file handler class"""

    __path = ''
    __traceFile = None
    __traceHeaderFun = None

    __TraceHeader = {}
    __headerLength = 0
    __traceNumber = -1
    __pointCount = -1
    __sampleCoding = -1
    __sampleLength = 0
    __cryptoDataLength = 0
    __titleSpace = 0
    __globalTraceTitle = 'SCAStudio'
    __description = None
    __xAxisOffset = 0
    __xLabel = ''
    __yLabel = ''
    __xAxisScale = 0
    __yAxisScale = 0
    __traceOffsetForDisp = 0
    __logScale = 0

    def __init__(self, path):
        self.__path = path
        self.__traceFile = CommonFile(path)
        self.__traceHeaderFun = {b'\x41': self.__NT, b'\x42': self.__NS, b'\x43': self.__SC, b'\x44': self.__DS,
                                 b'\x45': self.__TS,
                                 b'\x46': self.__GT, b'\x47': self.__DC, b'\x48': self.__XO, b'\x49': self.__XL,
                                 b'\x4A': self.__YL,
                                 b'\x4B': self.__XS, b'\x4C': self.__YS, b'\x4D': self.__TO, b'\x4E': self.__LS,
                                 b'\x5F': self.__TB,
                                 b'\x55': self.__UN, b'\x04': self.__UN, b'\x9A': self.__UN}

    @property
    def filePath(self):
        return self.__path

    @filePath.setter
    def filePath(self, value):
        self.__path = value

    @property
    def traceNumber(self):
        return self.__traceNumber

    @traceNumber.setter
    def traceNumber(self, value):
        self.__traceNumber = value

    @property
    def pointNumber(self):
        return self.__pointCount

    @pointNumber.setter
    def pointNumber(self, value):
        self.__pointCount = value

    @property
    def sampleCoding(self):
        return self.__sampleCoding

    @sampleCoding.setter
    def sampleCoding(self, value):
        self.__sampleCoding = value

    @property
    def sampleLength(self):
        return self.__sampleLength

    @sampleLength.setter
    def sampleLength(self, value):
        self.__sampleLength = value

    @property
    def cryptoDataCount(self):
        return self.__cryptoDataLength

    @cryptoDataCount.setter
    def cryptoDataCount(self, value):
        self.__cryptoDataLength = value

    @property
    def title_space(self):
        return self.__titleSpace

    @title_space.setter
    def title_space(self, value):
        self.__titleSpace = value

    @property
    def header_length(self):
        return self.__headerLength

    def parseFileHeader(self):
        logging.debug('Start Parsing Trace Header')
        self.__traceFile.openFile('rb')
        while True:
            ch = self.__traceFile.readbyte(1)
            logging.debug('Parsing Trace Header : ' + ch.hex())
            try:
                self.__traceHeaderFun[ch]()
            except KeyError:
                print("Key Error: Invalid Trs File Format.")
                break
            except ValueError:
                print("Value Error: Invalid Trs File Format.")
                break
            if ch == b'\x5F':
                logging.debug('Parsing Trace Header Finished')
                break
        self.__traceFile.closeFile()

    def generateTraceHeader(self):
        traceHeader = b'\x41\x04'
        traceHeader += self.__traceNumber.to_bytes(4, 'little')
        traceHeader += b'\x42\x04'
        traceHeader += self.__pointCount.to_bytes(4, 'little')
        traceHeader += b'\x43\x01'
        if self.__sampleCoding == 0:
            traceHeader += self.__sampleLength.to_bytes(1, 'little')
        else:
            traceHeader += (self.__sampleLength | 0x10).to_bytes(1, 'little')

        traceHeader += b'\x44\x02'
        traceHeader += self.__cryptoDataLength.to_bytes(2, 'little')
        traceHeader += b'\x5F\x00'

        self.__traceFile.openFile('wb')
        self.__traceFile.writeByte(traceHeader)
        self.__traceFile.closeFile()

    def generateTrace(self, point, cryptoData=None, title=None):
        traceStr = b''
        self.__traceFile.openFile('ab+')
        if title is not None:
            traceStr += title.encode('utf8')
            # self.__traceFile.writeByte(title.encode('utf8'))
        if cryptoData is not None:
            traceStr += bytes(cryptoData)
            # self.__traceFile.writeByte(bytes(cryptoData))
        if self.__sampleCoding == 0:
            if self.__sampleLength == 1:
                traceStr += bytes(point)
            elif self.__sampleLength == 2:
                for i in point:
                    traceStr += pack('<H', i)
            elif self.__sampleLength == 4:
                for i in point:
                    traceStr += pack('<I', i)
        else:
            # self.__traceFile.writeFile(point)
            traceStr += pack('<' + str(self.__pointCount) + 'f', *point)
            # for i in point:
            #     # traceStr += pack('<f', i)
            #     self.traceFile.writeByte(pack('<f', i))

        self.__traceFile.writeByte(traceStr)
        self.__traceFile.closeFile()

    def __NT(self):
        """0x41, NT, Number of traces"""
        data_length = self.__readHeaderDataLength()
        if data_length != 4:
            logging.error('Wrong trace header : NT')
            raise ValueError('Wrong Trace Header')
        self.__traceNumber = self.__traceFile.readint(data_length)
        logging.debug('Trace Number : ' + str(self.__traceNumber))

    def __NS(self):
        """0x42, NS, Number of samples per trace"""
        data_length = self.__readHeaderDataLength()
        if data_length != 4:
            logging.error('Wrong trace header : NS')
            raise ValueError('Wrong Trace Header')
        self.__pointCount = self.__traceFile.readint(data_length)
        logging.debug('Point Count : ' + str(self.__pointCount))

    def __SC(self):
        """0x43, SC, Sample Coding"""
        data_length = self.__readHeaderDataLength()
        if data_length != 1:
            logging.error('Wrong Trace Header : SC')
            raise ValueError('Wrong Trace Header')
        value_tmp = self.__traceFile.readint(1)
        self.__sampleCoding = (value_tmp & 0x10)
        self.__sampleLength = value_tmp & 0x0F
        logging.debug('Sample Coding : ' + str(self.__sampleCoding))
        logging.debug('Sample Length : ' + str(self.__sampleLength))

    def __DS(self):
        """0x44, DS, Length of cryptographic data included in trace"""
        data_length = self.__readHeaderDataLength()
        if data_length != 2:
            logging.error('Wrong Trace Header : TS')
            raise ValueError('Wrong Trace Header')
        self.__cryptoDataLength = self.__traceFile.readint(data_length)
        logging.debug('Crypto Data Length : ' + str(self.__cryptoDataLength))

    def __TS(self):
        """0x45, TS, Title space reserved per trace"""
        data_length = self.__readHeaderDataLength()
        if data_length != 1:
            logging.error('Wrong Trace Header : TS')
            raise ValueError('Wrong Trace Header')
        self.__titleSpace = self.__traceFile.readint(data_length)
        logging.debug('Title Space : ' + str(self.__titleSpace))

    def __GT(self):
        """0x46, GT, Global trace title"""
        data_length = self.__readHeaderDataLength()
        self.__globalTraceTitle = self.__traceFile.readstr(data_length)
        logging.debug('Global Trace Title : ' + self.__globalTraceTitle)

    def __DC(self):
        """0x47, DC, Description"""
        data_length = self.__readHeaderDataLength()
        self.__description = self.traceFile.__readstr(data_length)
        logging.debug('Description : ' + self.__description)

    def __XO(self):
        """0x48, XO, Offset in X-axis for trace representation"""
        data_length = self.__readHeaderDataLength()
        if data_length != 4:
            logging.error('Wrong Trace Header : XO')
            raise ValueError('Wrong Trace Header : XO')
        self.__xAxisOffset = self.__traceFile.readint()
        logging.debug('X-axis Offset : ' + str(self.__xAxisOffset))

    def __XL(self):
        """0x49, XL, Label of X-axis"""
        data_length = self.__readHeaderDataLength()
        self.__xLabel = self.__traceFile.readstr(data_length)
        logging.debug('X Label : ' + self.__xLabel)

    def __YL(self):
        """0x4A, YL, Label of Y-axis"""
        data_length = self.__readHeaderDataLength()
        self.__yLabel = self.__traceFile.readstr(data_length)
        logging.debug('Y Label : ' + self.__yLabel)

    def __XS(self):
        """0x4B, XS, Scale value for X-axis"""
        data_length = self.__readHeaderDataLength()
        if data_length != 4:
            logging.error('Wrong Trace Header : XS')
            raise ValueError
        self.__xAxisScale = self.__traceFile.readfloat(data_length)
        logging.debug('X-axis Scale : ' + str(self.__xAxisScale))

    def __YS(self):
        """0x4C, YS, Scale value for Y-axis"""
        data_length = self.__readHeaderDataLength()
        if data_length != 4:
            logging.error('Wrong Trace Header : YS')
            raise ValueError
        self.__yAxisScale = self.__traceFile.readfloat(data_length)
        logging.debug('Y-axis Scale : ' + str(self.__xAxisScale))

    def __TO(self):
        """0x4D, TO, Trace offset for displaying trace numbers"""
        data_length = self.__readHeaderDataLength()
        if data_length != 4:
            logging.error('Wrong Trace Header : TO')
            raise ValueError
        self.__traceOffsetForDisp = self.__traceFile.readint(data_length)
        logging.debug('Trace Offet For Displying : ' + self.__traceOffsetForDisp)

    def __LS(self):
        """0x4E, LS, Logarithmic scale"""
        data_length = self.__readHeaderDataLength()
        if data_length != 1:
            logging.error('Wrong Trace header : LS')
            raise ValueError
        self.__logScale = self.__traceFile.readint(1)
        logging.debug('Log Scale : ' + str(self.__logScale))

    def __TB(self):
        """0x5F, TB, Trace block marker: an empty TLV that marks the end of the header"""
        self.__readHeaderDataLength()
        self.__headerLength = self.__traceFile.byteNum
        logging.debug('Trace Header Length : ' + str(self.__headerLength))

    def __UN(self):
        """Unknown header"""
        pass

    def __readHeaderDataLength(self):
        data_length = self.__traceFile.readint(1)
        if data_length & 0x80:
            data_length &= 0x7F
            data_length = self.__traceFile.readint(data_length)
        return data_length

    def getTrace(self, index):
        if index < 0 or index > self.__traceNumber - 1:
            logging.error('Wrong Trace Index')
            raise ValueError('Wrong Trace Index')

        samplePoint = ()
        traceTitle = ''
        cryptoData = None
        self.__traceFile.openFile('rb')
        self.__traceFile.seekfile(self.__headerLength + index * (
                    self.__titleSpace + self.__cryptoDataLength + self.__pointCount * self.__sampleLength))
        if self.__titleSpace != 0:
            traceTitle = self.__traceFile.readstr(self.titleSpace).decode('utf-8')
            logging.debug('Trace %d title : %s' % (index, traceTitle))
        if self.__cryptoDataLength != 0:
            cryptoData = list(self.__traceFile.readbyte(self.__cryptoDataLength))
            logging.debug('CryptoData:' + str(cryptoData))
        if self.__pointCount != 0:
            if self.__sampleCoding == 0:
                bstr = self.__traceFile.readbyte(self.__sampleLength * self.__pointCount)
                # print(index)
                if self.__sampleLength == 1:
                    samplePoint = unpack(str(self.__pointCount) + 'B', bstr)
                elif self.sampleLength == 2:
                    samplePoint = unpack('<' + str(self.__pointCount) + 'H', bstr)
                elif self.sampleLength == 4:
                    samplePoint = unpack('<' + str(self.__pointCount) + 'I', bstr)
            else:
                bstr = self.__traceFile.readbyte(self.__sampleLength * self.__pointCount)
                samplePoint = unpack('<' + str(self.__pointCount) + 'f', bstr)

        self.__traceFile.closeFile()

        return [samplePoint, cryptoData, traceTitle]

    def get_trace_npy(self, index_range=None):
        if index_range is None:
            index_range = np.arange(0, self.traceNumber)
        trace_count = index_range.shape[0]
        sample_mat = np.zeros((trace_count, self.pointNumber))

        for index in range(trace_count):
            sample_mat[index, :], _, _ = self.getTrace(index_range[index])
            progress(index, trace_count, 'Extract Sample')

        return sample_mat

    def get_crypto_data_npy(self, index_range=None):
        if index_range is None:
            index_range = np.arange(0, self.traceNumber)
        trace_count = index_range.shape[0]
        crypto_data_mat = np.zeros((trace_count, self.cryptoDataCount), dtype=np.uint8)
        for index in range(trace_count):
            _, crypto_data_mat[index, :], _ = self.getTrace(index_range[index])
            progress(index, trace_count, 'Extract Crypto Data')
        return crypto_data_mat

    def __str__(self):
        return "cryptoDataCount = {0}\nsampleLength = {1}\nsampleCoding = {2}\n" \
        "pointNumber = {3}\ntraceNumber = {4}\ntraceFile = {5}".format(self.__cryptoDataLength,
                                                                              self.__sampleLength, self.__sampleCoding,
                                                                              self.__pointCount, self.__traceNumber,
                                                                              self.__path)



def trs2Npz(data_file_path,trs_name,npz_name,trace_num):
    '''
    该函数将trs文件转化为npz格式
    :param data_file_path: 目标文件位置 示例：data_file_path = r'G:\lowpass'
    :param trs_name: 曲线文件名 示例：trs_name = r'\attiny161433V2M + LowPass'
    :param npz_name: 需要保存的npz目标文件名 示例：npz_name = r'\attiny161433V2M'
    :param trace_num: 曲线条数 示例：trace_num=110000
    :return: null
    '''
    for dirpath, dirnames, filenames in os.walk(data_file_path):
        file_name = filenames
        break
    if npz_name + '.npz' in file_name:
        print(npz_name + '.npz已经存在')
        return
    th = TrsHandler(data_file_path + '\\' + trs_name + '.trs')
    th.parseFileHeader()

    # 获得trs文件中的能量迹，np.arange(20000)意思是获取0-19999曲线，get_trace_npy可任意获取任意索引的曲线
    trace_mat = th.get_trace_npy(np.arange(trace_num))

    # 获得trs文件中的明密文
    crypto_mat = th.get_crypto_data_npy(np.arange(trace_num))
    #　保存
    np.savez(data_file_path + '\\' + npz_name + '.npz', trace = trace_mat, crypto_data = crypto_mat[:,0])

def getIntermediateMatrix(data_file_path, npz_name):
    # # 计算中间值矩阵

    # 这里是示例，随机产生了明文，真正攻击时需要提取曲线中的明文

    # pt = np.random.randint(low=0, high=256, size=100000, dtype='uint8')

    # # Fixed key, 0x42
    # key = 0x42
    # # True Sbox In
    # sin = pt ^ key
    #target = np.load(r"G:\lowpass\attiny161433V2M")
    target = np.load(data_file_path + '\\' + npz_name + '.npz')
    pt = target["crypto_data"]
    # AES S盒
    sbox = [  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
      0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
      0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
      0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
      0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
      0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
      0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
      0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
      0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
      0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
      0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
      0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
      0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
      0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
      0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
      0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16]

    sbox = np.array(sbox, dtype='uint8')

    # 汉明重量矩阵，把256种值的汉明重量存为了数组，方便使用
    hwmat = np.load('hwMat.npy')
    hwmat = hwmat.reshape(-1)

    # 根据明文，遍历可能的key,计算可能的S盒输入输出
    guess_sin = []
    guess_sout = []
    for key_guess in range(256):
        # 明文异或猜测key
        tmp = pt ^ key_guess
        guess_sin.append(tmp)
        # 得到对应的猜测Ｓ盒输出
        tmp = sbox[tmp]
        guess_sout.append(tmp)

    guess_sin_value = np.transpose(np.vstack(guess_sin))
    guess_sin_hw = hwmat[guess_sin_value]

    #得到猜测Ｓ盒输出的汉明重量
    guess_sout_value = np.transpose(np.vstack(guess_sout))
    guess_sout_hw = hwmat[guess_sout_value]
    return [target,guess_sout_hw]