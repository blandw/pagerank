"""
PageRank

Tested on a graph with 5.7 million pages and 130 million links. Completed 100
iterations in less than 4 minutes (excluding file IO).

Input file format:

1. First line: the number of pages in the graph
2. Second line: the number of links in the graph
3. For each page, one line of the form:

         SOURCE, DEST1, DEST2, ...

   where SOURCE is the integer identifier of a page, and DEST1, DEST2, ... is
   a comma-delimited list of zero or more pages to which SOURCE links.

Pages must be numbered from 0 to (n - 1). Pages with no outgoing links do not
need to have a line in the file.

Example input file:

3
4
0, 1, 2
1, 2
2, 0

This represents a web with 3 pages. Page 0 links to pages 1 and 2.  Page 1
links to page 2. Page 2 links to page 0.
"""

import scipy.sparse as spm
import scipy.linalg as lin
import numpy as np
import csv
import logging
import sys

def pageRank(link_mat, dangle_vec, damp = 0.85, eps = 0.0000001, max_iters = 50):
    assert link_mat.shape[0] == link_mat.shape[1] == len(dangle_vec)
    assert 0 <= damp <= 1
    assert eps >= 0
    assert max_iters >= 0
    n = len(dangle_vec)
    u = np.ones(n, dtype = np.float32) / n # uniform vector with each element equal to (1/n)
    r = u
    r_old = u
    residual = eps + 1
    iters = 0
    while True:
        if residual <= eps:
            stop = 'residual'
            break
        elif iters == max_iters:
            stop = 'iters'
            break
        iters += 1
        logging.info('Iteration ' + str(iters))
        #
        r_old = np.copy(r)
        r = damp * np.dot(dangle_vec, r_old) * u
        r += (1 - damp) * u
        r += damp * (link_mat * r_old) # do link_mat * r_old first to avoid forming another (n x n) matrix
        r /= lin.norm(r, 1) # require |r| = 1
        #
        residual = lin.norm(r_old - r, 1)
        logging.debug('    R = ' + str(r))
        logging.info('    Residual = ' + str(residual))
    assert abs(1 - lin.norm(r, 1)) < 0.0001
    return r, residual, iters, stop

def readLinksFile(filename):
    with open(filename) as f:
        num_pages = int(f.next())
        num_links = int(f.next())
        logging.info(str(num_pages) + ' pages, ' + str(num_links) + ' links')
        sources = np.zeros(num_links, dtype = np.int32)
        links = np.zeros(num_links, dtype = np.int32)
        normadj = np.zeros(num_links, dtype = np.float32)
        dangle = np.ones(num_pages, dtype = np.int8)
        links_seen = 0
        for line in csv.reader(f):
            source = int(line[0])
            outdeg = len(line) - 1
            if outdeg > 0:
                dangle[source] = 0
                for i in range(0, outdeg):
                    sources[links_seen + i] = source
                    links[links_seen + i] = line[i + 1]
                    normadj[links_seen + i] = 1.0 / outdeg
                links_seen += outdeg
    assert links_seen == num_links
    link_mat = spm.coo_matrix((normadj, (links, sources)), shape = (num_pages, num_pages)).tocsr()
    return num_pages, link_mat, dangle

def writeRanksFile(filename, r):
    with open(filename, 'w') as f:
        for i in range(0, len(r)):
            f.write(str(i) + ', ' + str(r[i]) + '\n')

def main(argv = None):
    logging.basicConfig(level = logging.DEBUG)
    if argv is None:
        argv = sys.argv
    logging.info('Reading links file')
    links_filename = argv[1]
    n, link_mat, dangle_vec = readLinksFile(links_filename)
    logging.info('Computing PageRank')
    r, residual, iters, stop = pageRank(link_mat, dangle_vec)
    logging.info('Done (' + stop + ').')
    logging.debug('Page with highest PageRank: ' + str(r.argmax()) + ' (' + str(r.max()) + ')')
    logging.info('Writing PageRank file')
    writeRanksFile('ranks', r) # output file
    logging.info('Done.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
