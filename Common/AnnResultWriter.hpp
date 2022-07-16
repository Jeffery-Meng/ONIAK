#ifndef __ANNRESULT_HPP__
#define __ANNRESULT_HPP__

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>

#include "Exception.h"

namespace AnnResults {

// default header & format for external memory implementations
// query id, k id (id for the k-th nearest neighbor), distance for the k-th nearest neighbor,
// ground truth distance, query time (microseconds), total IOs
const char * _DEFAULT_HEADER_E_ = "#qid,#kid,#rid,rdist,gdist,ratio,qtime(us),#io";
const char * _DEFAULT_FMT_E_ = "iiiiiffi";
const char * _DEFAULT_HEADER_I_ = "#qid,#kid,#rid,rdist,gdist,ratio,qtime(us)";
const char * _DEFAULT_FMT_I_ = "iiiiiff";
} // namespace AnnResults

class AnnResultWriter {
public:
  AnnResultWriter(const std::string &filename, bool allowOverwrite = false)
      : _fp(nullptr), _filename_cp(filename) {
#ifdef DEBUG
    fprintf(stdout, "Trying to open file %s\n", _filename_cp.c_str());
#endif
    if (!allowOverwrite && _exists()) {
      throw npp::Exception("AnnResultWriter::AnnResultWriter(): file " +
                               _filename_cp + " already exists", __FILE__, __LINE__);
    }
    if ((_fp = fopen(_filename_cp.c_str(), "w")) == nullptr)
      throw std::runtime_error(
          "AnnResultWriter::AnnResultWriter(): Failed to open file " +
          _filename_cp);

#ifdef DEBUG
    fprintf(stdout, "Open file %s successfully\n", _filename_cp.c_str());
#endif
  }

  AnnResultWriter(const AnnResultWriter &) = delete;
  AnnResultWriter &operator=(const AnnResultWriter &) = delete;

  ~AnnResultWriter() {
#ifdef DEBUG
    fprintf(stdout, "Close file %s\n", _filename_cp.c_str());
#endif
    if (_fp)
      fclose(_fp);
  }

  bool writeRow(const char *fmt, ...) {
    assert(_fp != nullptr && "File MUST be opened");

    int count = 0;
    bool success = true;
    va_list args;
    va_start(args, fmt);

    while (*fmt != '\0') {
      switch (*fmt) {
      case 'i': {
        long long i = va_arg(args,  int);
        if (!_writeOne("%d", i, (count == 0)))
          success = false;
        break;
      }
      case 'f':
      case 'd': {
        double d = va_arg(args, double);
        if (!_writeOne("%.6f", d, (count == 0)))
          success = false;
        break;
      }
      case 'c': {
        int c = va_arg(args, int);
        if (!_writeOne("%c", c, (count == 0)))
          success = false;
        break;
      }
      case 's': {
        char *s = va_arg(args, char *);
        if (!_writeOne("%s", s, (count == 0)))
          success = false;
        break;
      }
      default:
        throw std::runtime_error("Unsupported format \'" +
                                 std::to_string(*fmt) + "\'\n");
      }
      if (!success)
        break;
      ++fmt;
      ++count;
    }

    va_end(args);
    fprintf(_fp, "\n");

    return success;
  }

private:
  template <typename AnyPrintableType>
  bool _writeOne(const char *format, const AnyPrintableType &val,
                 bool isFirst = false) {

    int r1 = fprintf(_fp, "%s", (isFirst ? "" : ","));
    int r2 = fprintf(_fp, format, val);

    auto success = (r1 >= 0 && r2 >= 0);
#ifdef DEBUG
    if (!success) {
      perror("AnnResultWriter::_wrietOne() failed.\nError: ");
    }

#endif
    return success;
  }
  bool _exists() const {
    FILE *fp = fopen(_filename_cp.c_str(), "r");
    bool ex = (fp != nullptr);
    if (ex)
      fclose(fp);

    return ex;
  }
  FILE *_fp;
  std::string _filename_cp;
};

class CppFileHelper {
  public:
  CppFileHelper(const std::string &filename): _file(filename) {
    if (!_file) {
      std::cerr << "Warning: " << filename << " is not opened!" << std::endl;
    }
  }
  // TODO: check if file exsits before opening it

  CppFileHelper(const CppFileHelper& ) = delete; 
  template<typename T>
  void print_one_line(T value) {
      _file << value << std::endl;
  }

  template<typename T, typename... Targs>
  void print_one_line(T value, Targs... Fargs) {
      _file << value << "\t";
      print_one_line(Fargs...);
  }

  template<typename T1, typename T2>
  void print_pair(const T1& t1, const T2& t2) {
    _file << std::make_pair(t1, t2) << "\t";
  }

  template<typename T>
  CppFileHelper& operator<<(const T& t) {
    _file << t;
    return *this;
  }

  private:
  std::ofstream _file;
};


class BinaryFileHelper {
public:
  BinaryFileHelper(const std::string &filename) {
    _file.open(filename, std::ios::binary);
    if (!_file) {
      std::cerr << "Warning: " << filename << " is not opened!" << std::endl;
    }
  }  

  BinaryFileHelper(const BinaryFileHelper&) = delete;

  template<typename T>
  BinaryFileHelper& operator<<(const T& t) {
    _file.write((char*) &t, sizeof(T));
    return *this;
  }

  template<typename T>
  BinaryFileHelper& operator<<(const std::vector<T>& t) {
    int sz = t.size();
    _file.write((char*) &sz, sizeof(sz));
    for (const auto& item : t) {
      _file.write((char*) &item, sizeof(T));
    }
    return *this;
  }

private:
  std::ofstream _file;
};


class BinaryFileReader {
public:
  BinaryFileReader(const std::string &filename) {
    _file.open(filename, std::ios::binary);
    if (!_file) {
      std::cerr << "Warning: " << filename << " is not opened!" << std::endl;
    }
  }  

  BinaryFileReader(const BinaryFileReader&) = delete;

  template<typename T>
  BinaryFileReader& operator>>(T& t) {
    _file.read((char*) &t, sizeof(T));
    return *this;
  }

  template<typename T>
  BinaryFileReader& operator>>(std::vector<T>& t) {
    int size;
    _file.read((char*) &size, sizeof(int));
    _file.read((char*) t.data(), size * sizeof(T));
    return *this;
  }

  template<typename T>
  BinaryFileReader& operator>>(std::vector<std::vector<T> >& t) {
    while(_file) {
    std::vector<T> buffer;
    int size;
    _file.read((char*) &size, sizeof(int));
    _file.read((char*) buffer.data(), size * sizeof(T));
    t.push_back(std::move(buffer));
    }
    return *this;
  }
private:
  std::ifstream _file;
};

#endif
