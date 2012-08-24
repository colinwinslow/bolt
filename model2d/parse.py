#!/usr/bin/env python
# coding: utf-8

import re
import os
import csv
import sys
import glob
import tempfile
import fileinput
import subprocess
from models import SentenceParse
from sqlalchemy.orm.exc import NoResultFound



def parse_sentences(ss, parser_path='../bllip-parser'):
    """parse sentences with the charniak parser"""
    # create a temporary file and write the sentences in it
    temp = tempfile.NamedTemporaryFile()
    for s in ss:
        temp.write('<s> %s </s>\n' % s)
    temp.flush()
    # where am i?
    prev_path = os.getcwd()
    # get into the charniak parser directory
    os.chdir(parser_path)
    # call the parser
    proc = subprocess.Popen(['./parse.sh', '-t4', temp.name],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    # capture output
    output = proc.communicate()[0]
    # return to where i was
    os.chdir(prev_path)
    # get rid of temporary file
    temp.close()
    # return the parse trees
    return output.splitlines()



def modify_parses(trees, tregex_path='stanford-tregex',
                         surgery_path='surgery'):
    """modify parse trees using tsurgeon"""
    # write trees to temp file
    temp = tempfile.NamedTemporaryFile()
    for t in trees:
        temp.write(t + '\n')
    temp.flush()

    # tregex jar location
    jar = os.path.join(tregex_path, 'stanford-tregex.jar')
    tsurgeon = 'edu.stanford.nlp.trees.tregex.tsurgeon.Tsurgeon'
    # surgery scripts
    surgery = sorted(glob.glob(os.path.join(surgery_path, '*')))
    proc = subprocess.Popen(['java', '-mx100m', '-cp', jar, tsurgeon,
                             '-s', '-treeFile', temp.name] + surgery,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    # get output
    output = proc.communicate()[0]
    # delete temp file
    temp.close()
    # return modified parse trees
    return output.splitlines()



def parse_generator_data(datafile):
    """parse a file with data generated by table2d generator"""
    xlocs, ylocs, sentences = [], [], []
    # this is how each observation looks like
    # for example: Vec2(5.31, 5.11); on the table
    pattern = r'^Vec2\((?P<x>-?[0-9.]+), (?P<y>-?[0-9.]+)\); (?P<sent>.+)$'
    for line in datafile:
        match = re.match(pattern, line)
        if match:
            xlocs.append(float(match.group('x')))
            ylocs.append(float(match.group('y')))
            sentences.append(match.group('sent'))
    return xlocs, ylocs, sentences



def get_modparse(sentence):
    """returns the modified parse tree for a sentence"""
    sp_db = SentenceParse.get_sentence_parse(sentence)

    try:
        res = sp_db.one()
        modparsetree = res.modified_parse
    except NoResultFound:
        parsetree = parse_sentences([sentence])[0]
        modparsetree = modify_parses([parsetree])[0]
        SentenceParse.add_sentence_parse(sentence, parsetree, modparsetree)

    return modparsetree



if __name__ == '__main__':
    # parse data from file or stdin
    xlocs, ylocs, sentences = parse_generator_data(fileinput.input())
    # parse sentences
    parses = parse_sentences(sentences)
    # write csv data to stdout
    writer = csv.writer(sys.stdout, lineterminator='\n')
    writer.writerow(['xloc', 'yloc', 'sentence', 'parse'])
    for row in zip(xlocs, ylocs, sentences, parses):
        writer.writerow(row)
