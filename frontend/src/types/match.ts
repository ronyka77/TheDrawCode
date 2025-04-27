
export interface Team {
  id: string;
  name: string;
  logoUrl: string;
  code: string;
}

export interface Prediction {
  homeWinProbability: number;
  drawProbability: number;
  awayWinProbability: number;
  status: 'ready' | 'pending' | 'error';
  lastUpdated?: string;
}

export interface Match {
  id: string;
  homeTeam: Team;
  awayTeam: Team;
  league: {
    id: string;
    name: string;
    country: string;
    logoUrl?: string;
  };
  startTime: string;
  status: 'scheduled' | 'live' | 'finished' | 'canceled';
  score?: {
    home: number;
    away: number;
  };
  prediction: Prediction;
}

export interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'analyst' | 'user' | 'guest';
  avatar?: string;
}
