// given an input string s and a pattern p, implement regular expression matching with support for '.' and '*'.
// '.' Matches any single character.
// '*' Matches zero or more of the preceding element.

function isMatch(s, p) {
    // s: string to match
    // p: pattern to match
    // return: boolean
    if (s.length === 0 && p.length === 0) {
        return true;
    }
    if (s.length === 0 && p.length !== 0) {
        return false;
    }
    if (s.length !== 0 && p.length === 0) {
        return false;
    }
    if (p.length === 1) {
        if (s.length === 1) {
            return s === p;
        }
        return false;
    }
    if (p[1] === '*') {
        if (s[0] === p[0] || p[0] === '.') {
            return isMatch(s.slice(1), p) || isMatch(s, p.slice(2));
        }
        return isMatch(s, p.slice(2));
    }
    if (s[0] === p[0] || p[0] === '.') {
        return isMatch(s.slice(1), p.slice(1));
    }
    return false;
}
